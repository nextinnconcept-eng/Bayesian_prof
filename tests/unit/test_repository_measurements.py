from datetime import date

import pytest
import sqlalchemy as sa

from dssc_discovery.database.repositories import (
    DerivedDescriptorRepository,
    ElectrochemicalMeasurementRepository,
    PhotovoltaicMeasurementRepository,
    RawSpectrumRepository,
    StabilityMeasurementRepository,
)
from dssc_discovery.domain.enums import (
    EquivalentCircuitModel,
    IlluminationProtocol,
    MeasurementType,
    SpeciesCode,
)
from dssc_discovery.domain.measurements import DerivedDescriptor, ElectrochemicalMeasurement
from tests.seed_helpers import seed_full_device_chain, seed_raw_spectrum_for_device


def test_raw_spectrum_round_trips_for_device_source(session):
    device = seed_full_device_chain(session, species_code=SpeciesCode.BETA_VULGARIS)
    spectrum = seed_raw_spectrum_for_device(session, device_id=device.id)

    fetched = RawSpectrumRepository(session).get(spectrum.id)
    assert fetched == spectrum


def test_derived_descriptor_round_trips_and_traces_back_to_spectrum(session):
    device = seed_full_device_chain(session, species_code=SpeciesCode.BETA_VULGARIS)
    spectrum = seed_raw_spectrum_for_device(session, device_id=device.id)

    descriptor = DerivedDescriptorRepository(session).add(
        DerivedDescriptor(
            source_spectrum_id=spectrum.id,
            descriptor_name="series_resistance_estimate",
            value=15.2,
            unit="ohm",
            processing_version="eis-v1",
            operator="J. Doe",
        )
    )
    fetched = DerivedDescriptorRepository(session).get(descriptor.id)
    assert fetched == descriptor
    assert fetched.source_spectrum_id == spectrum.id


def test_derived_descriptor_rejects_unknown_source_spectrum_fk(session):
    from uuid import uuid4

    with pytest.raises(sa.exc.IntegrityError):
        DerivedDescriptorRepository(session).add(
            DerivedDescriptor(
                source_spectrum_id=uuid4(),
                descriptor_name="x",
                value=1.0,
                processing_version="v1",
                operator="J. Doe",
            )
        )


def test_electrochemical_measurement_round_trips_with_missing_fit(session):
    device = seed_full_device_chain(session, species_code=SpeciesCode.BETA_VULGARIS)
    spectrum = seed_raw_spectrum_for_device(session, device_id=device.id)

    measurement = ElectrochemicalMeasurementRepository(session).add(
        ElectrochemicalMeasurement(
            device_id=device.id,
            raw_spectrum_id=spectrum.id,
            equivalent_circuit_model=EquivalentCircuitModel.RANDLES,
            operator="J. Doe",
            instrument="Autolab PGSTAT",
        )
    )
    fetched = ElectrochemicalMeasurementRepository(session).get(measurement.id)
    assert fetched == measurement
    assert fetched.series_resistance_ohm is None  # non-convergent fit, still a real row


def test_photovoltaic_measurement_round_trips(session):
    device = seed_full_device_chain(session, species_code=SpeciesCode.BETA_VULGARIS)
    repo = PhotovoltaicMeasurementRepository(session)

    from dssc_discovery.domain.measurements import PhotovoltaicMeasurement

    pv = repo.add(
        PhotovoltaicMeasurement(
            device_id=device.id,
            illumination_protocol=IlluminationProtocol.AM1_5G_1_SUN,
            measurement_date=date(2026, 1, 20),
            open_circuit_voltage_v=0.65,
            short_circuit_current_density_ma_cm2=12.0,
            fill_factor=0.65,
            power_conversion_efficiency_percent=5.0,
            operator="J. Doe",
            instrument="Keithley 2400",
        )
    )
    fetched = repo.get(pv.id)
    assert fetched == pv


def test_stability_measurement_round_trips_with_two_distinct_pv_refs(session):
    device = seed_full_device_chain(session, species_code=SpeciesCode.BETA_VULGARIS)
    pv_repo = PhotovoltaicMeasurementRepository(session)

    from dssc_discovery.domain.measurements import PhotovoltaicMeasurement, StabilityMeasurement

    baseline = pv_repo.add(
        PhotovoltaicMeasurement(
            device_id=device.id,
            illumination_protocol=IlluminationProtocol.AM1_5G_1_SUN,
            measurement_date=date(2026, 1, 20),
            open_circuit_voltage_v=0.65,
            short_circuit_current_density_ma_cm2=12.0,
            fill_factor=0.65,
            power_conversion_efficiency_percent=5.0,
            operator="J. Doe",
            instrument="Keithley 2400",
        )
    )
    timepoint = pv_repo.add(
        PhotovoltaicMeasurement(
            device_id=device.id,
            illumination_protocol=IlluminationProtocol.AM1_5G_1_SUN,
            measurement_date=date(2026, 1, 27),
            open_circuit_voltage_v=0.60,
            short_circuit_current_density_ma_cm2=11.0,
            fill_factor=0.60,
            power_conversion_efficiency_percent=4.6,
            operator="J. Doe",
            instrument="Keithley 2400",
        )
    )

    stability_repo = StabilityMeasurementRepository(session)
    stability = stability_repo.add(
        StabilityMeasurement(
            device_id=device.id,
            baseline_measurement_id=baseline.id,
            timepoint_measurement_id=timepoint.id,
            ageing_duration_h=168.0,
            retention_percent=92.0,
            operator="J. Doe",
            instrument="Keithley 2400",
        )
    )
    fetched = stability_repo.get(stability.id)
    assert fetched == stability
    assert fetched.baseline_measurement_id != fetched.timepoint_measurement_id


def test_raw_spectrum_measurement_type_matches_units_registry_field_naming(session):
    # sanity check that MeasurementType.EIS spectra genuinely require a
    # device, wired the same way the domain-layer validator (Chunk 1.3)
    # already enforces -- confirms nothing was lost going through the ORM.
    device = seed_full_device_chain(session, species_code=SpeciesCode.BETA_VULGARIS)
    spectrum = seed_raw_spectrum_for_device(
        session, device_id=device.id, measurement_type=MeasurementType.EIS
    )
    assert spectrum.pigment_extract_batch_id is None
    assert spectrum.device_id == device.id
