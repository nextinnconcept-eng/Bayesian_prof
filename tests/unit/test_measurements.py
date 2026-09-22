from datetime import date, timedelta
from uuid import uuid4

import pytest
from pydantic import ValidationError

from dssc_discovery.domain.enums import (
    EquivalentCircuitModel,
    IlluminationProtocol,
    MeasurementType,
    MissingValueReason,
)
from dssc_discovery.domain.measurements import (
    DerivedDescriptor,
    ElectrochemicalMeasurement,
    PhotovoltaicMeasurement,
    RawSpectrum,
    StabilityMeasurement,
)

# --- RawSpectrum ---------------------------------------------------------


def _raw_spectrum_kwargs(**overrides):
    kwargs = dict(
        measurement_type=MeasurementType.UV_VIS,
        pigment_extract_batch_id=uuid4(),
        raw_data_reference="s3://raw/uvvis/001.csv",
        acquisition_date=date(2026, 1, 15),
        operator="J. Doe",
        instrument="Shimadzu UV-1800",
    )
    kwargs.update(overrides)
    return kwargs


def test_valid_chemical_optical_spectrum_construction():
    spectrum = RawSpectrum(**_raw_spectrum_kwargs())
    assert spectrum.measurement_type is MeasurementType.UV_VIS


def test_valid_device_spectrum_construction():
    spectrum = RawSpectrum(
        **_raw_spectrum_kwargs(
            measurement_type=MeasurementType.EIS,
            pigment_extract_batch_id=None,
            device_id=uuid4(),
        )
    )
    assert spectrum.device_id is not None


def test_raw_spectrum_round_trips_through_dict():
    spectrum = RawSpectrum(**_raw_spectrum_kwargs())
    restored = RawSpectrum.model_validate(spectrum.model_dump())
    assert restored == spectrum


def test_raw_spectrum_rejects_both_sources_set():
    with pytest.raises(ValidationError, match="exactly one"):
        RawSpectrum(**_raw_spectrum_kwargs(device_id=uuid4()))  # batch already set


def test_raw_spectrum_rejects_neither_source_set():
    with pytest.raises(ValidationError, match="exactly one"):
        RawSpectrum(**_raw_spectrum_kwargs(pigment_extract_batch_id=None))


def test_raw_spectrum_rejects_mismatched_source_for_measurement_type():
    # EIS is a device-type modality; pointing it at a pigment batch instead
    # of a device must be rejected.
    with pytest.raises(ValidationError, match="device_id"):
        RawSpectrum(
            **_raw_spectrum_kwargs(
                measurement_type=MeasurementType.EIS,
                pigment_extract_batch_id=uuid4(),
            )
        )


def test_raw_spectrum_rejects_future_acquisition_date():
    future_date = date.today() + timedelta(days=1)
    with pytest.raises(ValidationError, match="future"):
        RawSpectrum(**_raw_spectrum_kwargs(acquisition_date=future_date))


# --- DerivedDescriptor ----------------------------------------------------


def test_valid_derived_descriptor_construction():
    descriptor = DerivedDescriptor(
        source_spectrum_id=uuid4(),
        descriptor_name="lambda_max",
        value=520.0,
        unit="nm",
        processing_version="uvvis-v1",
        operator="J. Doe",
    )
    assert descriptor.descriptor_name == "lambda_max"


def test_derived_descriptor_round_trips_through_dict():
    descriptor = DerivedDescriptor(
        source_spectrum_id=uuid4(),
        descriptor_name="lambda_max",
        value=520.0,
        processing_version="uvvis-v1",
        operator="J. Doe",
    )
    restored = DerivedDescriptor.model_validate(descriptor.model_dump())
    assert restored == descriptor


def test_derived_descriptor_rejects_missing_source_spectrum_id():
    with pytest.raises(ValidationError):
        DerivedDescriptor(
            descriptor_name="lambda_max",
            value=520.0,
            processing_version="uvvis-v1",
            operator="J. Doe",
        )


def test_derived_descriptor_rejects_missing_processing_version():
    with pytest.raises(ValidationError):
        DerivedDescriptor(
            source_spectrum_id=uuid4(),
            descriptor_name="lambda_max",
            value=520.0,
            operator="J. Doe",
        )


# --- ElectrochemicalMeasurement -------------------------------------------


def test_valid_electrochemical_measurement_construction():
    measurement = ElectrochemicalMeasurement(
        device_id=uuid4(),
        raw_spectrum_id=uuid4(),
        equivalent_circuit_model=EquivalentCircuitModel.RANDLES,
        series_resistance_ohm=15.0,
        charge_transfer_resistance_ohm=200.0,
        recombination_resistance_ohm=500.0,
        electron_lifetime_ms=12.0,
        fit_r_squared=0.98,
        operator="J. Doe",
        instrument="Autolab PGSTAT",
    )
    assert measurement.fit_r_squared == 0.98


def test_electrochemical_measurement_allows_missing_fit_with_reason():
    # Non-convergent fit: result fields stay None, but the attempt is
    # still recorded rather than silently dropped.
    measurement = ElectrochemicalMeasurement(
        device_id=uuid4(),
        raw_spectrum_id=uuid4(),
        equivalent_circuit_model=EquivalentCircuitModel.RANDLES,
        missing_value_reason=MissingValueReason.INSTRUMENT_FAILURE,
        operator="J. Doe",
        instrument="Autolab PGSTAT",
    )
    assert measurement.series_resistance_ohm is None
    assert measurement.missing_value_reason is MissingValueReason.INSTRUMENT_FAILURE


def test_electrochemical_measurement_rejects_negative_resistance():
    with pytest.raises(ValidationError):
        ElectrochemicalMeasurement(
            device_id=uuid4(),
            raw_spectrum_id=uuid4(),
            equivalent_circuit_model=EquivalentCircuitModel.RANDLES,
            series_resistance_ohm=-5.0,
            operator="J. Doe",
            instrument="Autolab PGSTAT",
        )


# --- PhotovoltaicMeasurement -----------------------------------------------


def _pv_kwargs(**overrides):
    kwargs = dict(
        device_id=uuid4(),
        illumination_protocol=IlluminationProtocol.AM1_5G_1_SUN,
        measurement_date=date(2026, 1, 20),
        open_circuit_voltage_v=0.65,
        short_circuit_current_density_ma_cm2=12.0,
        fill_factor=0.65,
        power_conversion_efficiency_percent=5.0,
        operator="J. Doe",
        instrument="Keithley 2400",
    )
    kwargs.update(overrides)
    return kwargs


def test_valid_pv_measurement_construction():
    pv = PhotovoltaicMeasurement(**_pv_kwargs())
    assert pv.power_conversion_efficiency_percent == 5.0


def test_pv_measurement_round_trips_through_dict():
    pv = PhotovoltaicMeasurement(**_pv_kwargs())
    restored = PhotovoltaicMeasurement.model_validate(pv.model_dump())
    assert restored == pv


def test_pv_measurement_rejects_negative_pce():
    with pytest.raises(ValidationError):
        PhotovoltaicMeasurement(**_pv_kwargs(power_conversion_efficiency_percent=-1.0))


@pytest.mark.parametrize("bad_ff", [-0.1, 1.1])
def test_pv_measurement_rejects_fill_factor_outside_unit_interval(bad_ff):
    with pytest.raises(ValidationError):
        PhotovoltaicMeasurement(**_pv_kwargs(fill_factor=bad_ff))


def test_pv_measurement_rejects_pce_over_100():
    with pytest.raises(ValidationError):
        PhotovoltaicMeasurement(**_pv_kwargs(power_conversion_efficiency_percent=150.0))


# --- StabilityMeasurement ---------------------------------------------------


def _stability_kwargs(**overrides):
    kwargs = dict(
        device_id=uuid4(),
        baseline_measurement_id=uuid4(),
        timepoint_measurement_id=uuid4(),
        ageing_duration_h=168.0,
        retention_percent=92.0,
        operator="J. Doe",
        instrument="Keithley 2400",
    )
    kwargs.update(overrides)
    return kwargs


def test_valid_stability_measurement_construction():
    stability = StabilityMeasurement(**_stability_kwargs())
    assert stability.retention_percent == 92.0


def test_stability_measurement_round_trips_through_dict():
    stability = StabilityMeasurement(**_stability_kwargs())
    restored = StabilityMeasurement.model_validate(stability.model_dump())
    assert restored == stability


def test_stability_measurement_rejects_negative_retention():
    with pytest.raises(ValidationError):
        StabilityMeasurement(**_stability_kwargs(retention_percent=-1.0))


def test_stability_measurement_rejects_implausibly_high_retention():
    with pytest.raises(ValidationError):
        StabilityMeasurement(**_stability_kwargs(retention_percent=1050.0))


def test_stability_measurement_links_baseline_and_timepoint_separately():
    baseline_id = uuid4()
    timepoint_id = uuid4()
    stability = StabilityMeasurement(
        **_stability_kwargs(
            baseline_measurement_id=baseline_id, timepoint_measurement_id=timepoint_id
        )
    )
    assert stability.baseline_measurement_id == baseline_id
    assert stability.timepoint_measurement_id == timepoint_id
    assert stability.baseline_measurement_id != stability.timepoint_measurement_id
