"""Tests for cross-entity validation rules (Chunk 1.6)."""

from __future__ import annotations

from datetime import date
from uuid import uuid4

import pytest

from dssc_discovery.database.repositories import ExperimentalConditionRepository
from dssc_discovery.domain.enums import (
    DesignSource,
    EquivalentCircuitModel,
    IlluminationProtocol,
    MissingValueReason,
    SolventType,
    SpeciesCode,
    SurfaceTreatment,
)
from dssc_discovery.domain.experimental_condition import (
    ExperimentalCondition,
    ExperimentalConditionVector,
)
from dssc_discovery.domain.measurements import (
    ElectrochemicalMeasurement,
    PhotovoltaicMeasurement,
    StabilityMeasurement,
)
from dssc_discovery.validation.exceptions import (
    DuplicateConditionError,
    MissingProvenanceError,
    PhysicallyImplausibleError,
)
from dssc_discovery.validation.rules import (
    check_duplicate_condition,
    check_missing_value_reason,
    check_pce_consistency,
    check_stability_retention_consistency,
)
from tests.seed_helpers import seed_experimental_condition, seed_species


def _default_vector(**overrides: object) -> ExperimentalConditionVector:
    kwargs = dict(
        solvent=SolventType.ETHANOL,
        solvent_ratio=0.7,
        ph=5.5,
        extraction_temperature_c=25.0,
        extraction_time_min=60.0,
        pigment_concentration_mm=10.0,
        adsorption_time_min=120.0,
        sensitization_concentration_mm=0.5,
        sensitization_temperature_c=25.0,
        tio2_thickness_um=12.0,
        sintering_temperature_c=450.0,
        sintering_time_min=30.0,
        surface_treatment=SurfaceTreatment.NONE,
    )
    kwargs.update(overrides)
    return ExperimentalConditionVector(**kwargs)


# --- check_duplicate_condition ---------------------------------------------


def test_duplicate_condition_detected_even_with_different_condition_code(session):
    seed_species(session, SpeciesCode.BETA_VULGARIS)
    seed_experimental_condition(session, SpeciesCode.BETA_VULGARIS, condition_code="BV-COND-001")

    repo = ExperimentalConditionRepository(session)
    near_duplicate = ExperimentalCondition(
        species_code=SpeciesCode.BETA_VULGARIS,
        condition_code="BV-COND-DUPLICATE",  # different code, same recipe
        design_source=DesignSource.MANUAL,
        target_parameters=_default_vector(),
        operator="J. Doe",
    )
    with pytest.raises(DuplicateConditionError):
        check_duplicate_condition(repo, near_duplicate)


def test_different_species_is_not_flagged_as_duplicate(session):
    seed_species(session, SpeciesCode.BETA_VULGARIS)
    seed_species(session, SpeciesCode.CHROMOLAENA_ODORATA)
    seed_experimental_condition(session, SpeciesCode.BETA_VULGARIS, condition_code="BV-COND-001")

    repo = ExperimentalConditionRepository(session)
    different_species = ExperimentalCondition(
        species_code=SpeciesCode.CHROMOLAENA_ODORATA,
        condition_code="CO-COND-001",
        design_source=DesignSource.MANUAL,
        target_parameters=_default_vector(),
        operator="J. Doe",
    )
    check_duplicate_condition(repo, different_species)  # must not raise


def test_meaningfully_different_parameters_not_flagged_as_duplicate(session):
    seed_species(session, SpeciesCode.BETA_VULGARIS)
    seed_experimental_condition(session, SpeciesCode.BETA_VULGARIS, condition_code="BV-COND-001")

    repo = ExperimentalConditionRepository(session)
    different_recipe = ExperimentalCondition(
        species_code=SpeciesCode.BETA_VULGARIS,
        condition_code="BV-COND-002",
        design_source=DesignSource.MANUAL,
        target_parameters=_default_vector(ph=7.0),  # meaningfully different
        operator="J. Doe",
    )
    check_duplicate_condition(repo, different_recipe)  # must not raise


def test_condition_does_not_flag_itself_as_duplicate(session):
    seed_species(session, SpeciesCode.BETA_VULGARIS)
    condition = seed_experimental_condition(
        session, SpeciesCode.BETA_VULGARIS, condition_code="BV-COND-001"
    )
    repo = ExperimentalConditionRepository(session)
    check_duplicate_condition(repo, condition)  # comparing against itself must not raise


# --- check_pce_consistency --------------------------------------------------


def _pv_kwargs(**overrides: object) -> dict:
    kwargs = dict(
        device_id=uuid4(),
        illumination_protocol=IlluminationProtocol.AM1_5G_1_SUN,
        measurement_date=date(2026, 1, 20),
        open_circuit_voltage_v=0.65,
        short_circuit_current_density_ma_cm2=12.0,
        fill_factor=0.65,
        power_conversion_efficiency_percent=5.07,  # = 0.65*12.0*0.65/100*100
        operator="J. Doe",
        instrument="Keithley 2400",
    )
    kwargs.update(overrides)
    return kwargs


def test_consistent_pce_passes():
    pv = PhotovoltaicMeasurement(**_pv_kwargs())
    check_pce_consistency(pv)  # must not raise


def test_grossly_inconsistent_pce_raises():
    pv = PhotovoltaicMeasurement(**_pv_kwargs(power_conversion_efficiency_percent=50.0))
    with pytest.raises(PhysicallyImplausibleError):
        check_pce_consistency(pv)


def test_dark_protocol_skips_pce_check():
    pv = PhotovoltaicMeasurement(
        **_pv_kwargs(
            illumination_protocol=IlluminationProtocol.DARK,
            power_conversion_efficiency_percent=99.0,  # nonsense, but should be skipped
        )
    )
    check_pce_consistency(pv)  # must not raise -- no defined power density for DARK


# --- check_stability_retention_consistency ----------------------------------


def test_consistent_retention_passes():
    baseline = PhotovoltaicMeasurement(**_pv_kwargs(power_conversion_efficiency_percent=5.0))
    timepoint = PhotovoltaicMeasurement(**_pv_kwargs(power_conversion_efficiency_percent=4.6))
    stability = StabilityMeasurement(
        device_id=baseline.device_id,
        baseline_measurement_id=baseline.id,
        timepoint_measurement_id=timepoint.id,
        ageing_duration_h=168.0,
        retention_percent=92.0,  # 4.6/5.0*100
        operator="J. Doe",
        instrument="Keithley 2400",
    )
    check_stability_retention_consistency(stability, baseline, timepoint)  # must not raise


def test_inconsistent_retention_raises():
    baseline = PhotovoltaicMeasurement(**_pv_kwargs(power_conversion_efficiency_percent=5.0))
    timepoint = PhotovoltaicMeasurement(**_pv_kwargs(power_conversion_efficiency_percent=4.6))
    stability = StabilityMeasurement(
        device_id=baseline.device_id,
        baseline_measurement_id=baseline.id,
        timepoint_measurement_id=timepoint.id,
        ageing_duration_h=168.0,
        retention_percent=50.0,  # nowhere near 92%
        operator="J. Doe",
        instrument="Keithley 2400",
    )
    with pytest.raises(PhysicallyImplausibleError):
        check_stability_retention_consistency(stability, baseline, timepoint)


def test_zero_baseline_pce_skips_retention_check():
    baseline = PhotovoltaicMeasurement(**_pv_kwargs(power_conversion_efficiency_percent=0.0))
    timepoint = PhotovoltaicMeasurement(**_pv_kwargs(power_conversion_efficiency_percent=4.6))
    stability = StabilityMeasurement(
        device_id=baseline.device_id,
        baseline_measurement_id=baseline.id,
        timepoint_measurement_id=timepoint.id,
        ageing_duration_h=168.0,
        retention_percent=149.0,  # nonsense relative to any real ratio, but should be skipped
        operator="J. Doe",
        instrument="Keithley 2400",
    )
    check_stability_retention_consistency(stability, baseline, timepoint)  # must not raise


# --- check_missing_value_reason ---------------------------------------------

_EIS_FIELDS = [
    "series_resistance_ohm",
    "charge_transfer_resistance_ohm",
    "recombination_resistance_ohm",
    "electron_lifetime_ms",
]


def test_all_present_passes_with_no_reason():
    measurement = ElectrochemicalMeasurement(
        device_id=uuid4(),
        raw_spectrum_id=uuid4(),
        equivalent_circuit_model=EquivalentCircuitModel.RANDLES,
        series_resistance_ohm=15.0,
        charge_transfer_resistance_ohm=200.0,
        recombination_resistance_ohm=500.0,
        electron_lifetime_ms=12.0,
        operator="J. Doe",
        instrument="Autolab PGSTAT",
    )
    check_missing_value_reason(measurement, _EIS_FIELDS)  # must not raise


def test_missing_field_with_reason_passes():
    measurement = ElectrochemicalMeasurement(
        device_id=uuid4(),
        raw_spectrum_id=uuid4(),
        equivalent_circuit_model=EquivalentCircuitModel.RANDLES,
        missing_value_reason=MissingValueReason.INSTRUMENT_FAILURE,
        operator="J. Doe",
        instrument="Autolab PGSTAT",
    )
    check_missing_value_reason(measurement, _EIS_FIELDS)  # must not raise


def test_missing_field_without_reason_raises():
    measurement = ElectrochemicalMeasurement(
        device_id=uuid4(),
        raw_spectrum_id=uuid4(),
        equivalent_circuit_model=EquivalentCircuitModel.RANDLES,
        operator="J. Doe",
        instrument="Autolab PGSTAT",
    )
    with pytest.raises(MissingProvenanceError):
        check_missing_value_reason(measurement, _EIS_FIELDS)
