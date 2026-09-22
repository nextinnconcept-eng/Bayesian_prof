"""Tests for ML dataset assembly (Chunk 3.2): row selection (only
measured devices), group_id correctness for leakage-safe CV, missing
characterization data staying NaN (never silently zero), and
reproducibility given a fixed DB state.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pytest

from dssc_discovery.database.repositories import (
    DerivedDescriptorRepository,
    ElectrochemicalMeasurementRepository,
    PhotovoltaicMeasurementRepository,
    RawSpectrumRepository,
)
from dssc_discovery.domain.enums import (
    EquivalentCircuitModel,
    IlluminationProtocol,
    MeasurementType,
    SpeciesCode,
)
from dssc_discovery.domain.measurements import (
    DerivedDescriptor,
    ElectrochemicalMeasurement,
    PhotovoltaicMeasurement,
    RawSpectrum,
)
from dssc_discovery.features.dataset_builder import DATASET_COLUMNS, build_ml_dataset
from tests.seed_helpers import (
    seed_device,
    seed_experimental_condition,
    seed_photoanode_fabrication_batch,
    seed_pigment_extract_batch,
    seed_plant_sample,
    seed_sensitization_run,
    seed_species,
)


def _add_pv_measurement(session, device_id, **overrides):
    kwargs = dict(
        device_id=device_id,
        illumination_protocol=IlluminationProtocol.AM1_5G_1_SUN,
        measurement_date=date(2026, 1, 20),
        open_circuit_voltage_v=0.65,
        short_circuit_current_density_ma_cm2=12.0,
        fill_factor=0.65,
        power_conversion_efficiency_percent=5.07,
        operator="J. Doe",
        instrument="Keithley 2400",
    )
    kwargs.update(overrides)
    return PhotovoltaicMeasurementRepository(session).add(PhotovoltaicMeasurement(**kwargs))


def _build_condition_and_devices(
    session, species_code, condition_code, n_devices, batch_code_prefix
):
    seed_species(session, species_code)
    condition = seed_experimental_condition(session, species_code, condition_code=condition_code)
    devices = []
    for i in range(1, n_devices + 1):
        fab_batch = seed_photoanode_fabrication_batch(
            session, batch_code=f"{batch_code_prefix}-FAB-{i}"
        )
        device = seed_device(
            session,
            experimental_condition_id=condition.id,
            photoanode_fabrication_batch_id=fab_batch.id,
            device_code=f"{condition_code}-R{i}",
            replicate_number=i,
        )
        devices.append(device)
    return condition, devices


# --- row selection: only measured devices ------------------------------------


def test_empty_database_returns_empty_dataframe_with_correct_columns(session):
    df = build_ml_dataset(session)
    assert df.empty
    assert list(df.columns) == list(DATASET_COLUMNS)


def test_unmeasured_device_is_excluded(session):
    _build_condition_and_devices(session, SpeciesCode.BETA_VULGARIS, "BV-COND-001", 1, "BV")
    session.commit()

    df = build_ml_dataset(session)
    assert df.empty


def test_measured_device_is_included_with_exact_row_count(session):
    _condition, devices = _build_condition_and_devices(
        session, SpeciesCode.BETA_VULGARIS, "BV-COND-001", 2, "BV"
    )
    _add_pv_measurement(session, devices[0].id)  # only the first device is measured
    session.commit()

    df = build_ml_dataset(session)
    assert len(df) == 1
    assert df.iloc[0]["device_code"] == devices[0].device_code


def test_multiple_pv_measurements_use_the_earliest_as_primary(session):
    _condition, devices = _build_condition_and_devices(
        session, SpeciesCode.BETA_VULGARIS, "BV-COND-001", 1, "BV"
    )
    device = devices[0]
    _add_pv_measurement(
        session,
        device.id,
        measurement_date=date(2026, 1, 27),
        power_conversion_efficiency_percent=4.0,  # later re-scan, must NOT be used
    )
    _add_pv_measurement(
        session,
        device.id,
        measurement_date=date(2026, 1, 20),
        power_conversion_efficiency_percent=5.07,  # earliest, must be used
    )
    session.commit()

    df = build_ml_dataset(session)
    assert len(df) == 1
    assert df.iloc[0]["power_conversion_efficiency_percent"] == pytest.approx(5.07)


# --- group_id correctness -----------------------------------------------------


def test_group_id_shared_across_replicates_of_the_same_condition(session):
    condition, devices = _build_condition_and_devices(
        session, SpeciesCode.BETA_VULGARIS, "BV-COND-001", 3, "BV"
    )
    for device in devices:
        _add_pv_measurement(session, device.id)
    session.commit()

    df = build_ml_dataset(session)
    assert len(df) == 3
    assert df["group_id"].nunique() == 1
    assert df["group_id"].iloc[0] == str(condition.id)


def test_different_conditions_get_different_group_ids(session):
    _c1, devices1 = _build_condition_and_devices(
        session, SpeciesCode.BETA_VULGARIS, "BV-COND-001", 1, "BV"
    )
    _c2, devices2 = _build_condition_and_devices(
        session, SpeciesCode.CHROMOLAENA_ODORATA, "CO-COND-001", 1, "CO"
    )
    _add_pv_measurement(session, devices1[0].id)
    _add_pv_measurement(session, devices2[0].id)
    session.commit()

    df = build_ml_dataset(session)
    assert df["group_id"].nunique() == 2


# --- missing characterization data stays NaN, never 0 ------------------------


def test_missing_characterization_data_is_nan_not_zero(session):
    _condition, devices = _build_condition_and_devices(
        session, SpeciesCode.BETA_VULGARIS, "BV-COND-001", 1, "BV"
    )
    _add_pv_measurement(session, devices[0].id)
    session.commit()

    df = build_ml_dataset(session)
    row = df.iloc[0]
    for column in (
        "extraction_yield_percent",
        "lambda_max_nm",
        "bandwidth_nm",
        "integrated_absorption",
        "series_resistance_ohm",
        "charge_transfer_resistance_ohm",
        "recombination_resistance_ohm",
        "electron_lifetime_ms",
    ):
        assert np.isnan(row[column]), f"{column} should be NaN, got {row[column]}"


def test_populated_characterization_data_appears_correctly(session):
    seed_species(session, SpeciesCode.BETA_VULGARIS)
    condition = seed_experimental_condition(session, SpeciesCode.BETA_VULGARIS)
    sample = seed_plant_sample(session, SpeciesCode.BETA_VULGARIS)
    pigment_batch = seed_pigment_extract_batch(
        session, plant_sample_id=sample.id, extraction_yield_percent=42.5
    )
    fab_batch = seed_photoanode_fabrication_batch(session)
    device = seed_device(
        session,
        experimental_condition_id=condition.id,
        photoanode_fabrication_batch_id=fab_batch.id,
    )
    seed_sensitization_run(session, device_id=device.id, pigment_extract_batch_id=pigment_batch.id)
    session.commit()

    uvvis_spectrum = RawSpectrumRepository(session).add(
        RawSpectrum(
            measurement_type=MeasurementType.UV_VIS,
            pigment_extract_batch_id=pigment_batch.id,
            raw_data_reference='{"path": "x", "sha256": "y"}',
            acquisition_date=date(2026, 1, 15),
            operator="J. Doe",
            instrument="Shimadzu UV-1800",
        )
    )
    for name, value, unit in (
        ("lambda_max_nm", 520.0, "nm"),
        ("bandwidth_nm", 35.3, "nm"),
        ("integrated_absorption", 32.0, "AU*nm"),
    ):
        DerivedDescriptorRepository(session).add(
            DerivedDescriptor(
                source_spectrum_id=uvvis_spectrum.id,
                descriptor_name=name,
                value=value,
                unit=unit,
                processing_version="uvvis-v1",
                operator="J. Doe",
            )
        )

    ElectrochemicalMeasurementRepository(session).add(
        ElectrochemicalMeasurement(
            device_id=device.id,
            raw_spectrum_id=uvvis_spectrum.id,  # reused only as a valid FK here, not real EIS data
            equivalent_circuit_model=EquivalentCircuitModel.RANDLES,
            series_resistance_ohm=15.0,
            charge_transfer_resistance_ohm=50.0,
            recombination_resistance_ohm=300.0,
            electron_lifetime_ms=15.0,
            fit_r_squared=0.999,
            operator="J. Doe",
            instrument="Autolab PGSTAT",
        )
    )
    _add_pv_measurement(session, device.id)
    session.commit()

    df = build_ml_dataset(session)
    row = df.iloc[0]
    assert row["extraction_yield_percent"] == pytest.approx(42.5)
    assert row["lambda_max_nm"] == pytest.approx(520.0)
    assert row["bandwidth_nm"] == pytest.approx(35.3)
    assert row["integrated_absorption"] == pytest.approx(32.0)
    assert row["series_resistance_ohm"] == pytest.approx(15.0)
    assert row["electron_lifetime_ms"] == pytest.approx(15.0)


def test_multiple_eis_measurements_use_the_most_recent(session):
    seed_species(session, SpeciesCode.BETA_VULGARIS)
    condition = seed_experimental_condition(session, SpeciesCode.BETA_VULGARIS)
    fab_batch = seed_photoanode_fabrication_batch(session)
    device = seed_device(
        session,
        experimental_condition_id=condition.id,
        photoanode_fabrication_batch_id=fab_batch.id,
    )
    _add_pv_measurement(session, device.id)
    session.commit()

    raw_spectrum = RawSpectrumRepository(session).add(
        RawSpectrum(
            measurement_type=MeasurementType.EIS,
            device_id=device.id,
            raw_data_reference='{"path": "x", "sha256": "y"}',
            acquisition_date=date(2026, 1, 20),
            operator="J. Doe",
            instrument="Autolab PGSTAT",
        )
    )
    session.commit()

    eis_repo = ElectrochemicalMeasurementRepository(session)
    older = eis_repo.add(
        ElectrochemicalMeasurement(
            device_id=device.id,
            raw_spectrum_id=raw_spectrum.id,
            equivalent_circuit_model=EquivalentCircuitModel.RANDLES,
            series_resistance_ohm=99.0,  # stale value -- must NOT appear in the result
            operator="J. Doe",
            instrument="Autolab PGSTAT",
        )
    )
    session.commit()
    newer = eis_repo.add(
        ElectrochemicalMeasurement(
            device_id=device.id,
            raw_spectrum_id=raw_spectrum.id,
            equivalent_circuit_model=EquivalentCircuitModel.RANDLES,
            series_resistance_ohm=15.0,  # most recent -- must be the one used
            operator="J. Doe",
            instrument="Autolab PGSTAT",
        )
    )
    session.commit()
    assert older.created_at <= newer.created_at

    df = build_ml_dataset(session)
    assert df.iloc[0]["series_resistance_ohm"] == pytest.approx(15.0)


# --- reproducibility -----------------------------------------------------------


def test_reproducible_given_the_same_db_state(session):
    condition, devices = _build_condition_and_devices(
        session, SpeciesCode.BETA_VULGARIS, "BV-COND-001", 2, "BV"
    )
    for device in devices:
        _add_pv_measurement(session, device.id)
    session.commit()

    first = build_ml_dataset(session)
    second = build_ml_dataset(session)
    assert first.equals(second)


def test_row_order_is_deterministic_by_device_code(session):
    _condition, devices = _build_condition_and_devices(
        session, SpeciesCode.BETA_VULGARIS, "BV-COND-001", 3, "BV"
    )
    # add measurements out of device-creation order
    for device in reversed(devices):
        _add_pv_measurement(session, device.id)
    session.commit()

    df = build_ml_dataset(session)
    assert list(df["device_code"]) == sorted(df["device_code"])


def test_is_synthetic_flag_is_preserved(session):
    seed_species(session, SpeciesCode.BETA_VULGARIS)
    condition = seed_experimental_condition(session, SpeciesCode.BETA_VULGARIS)
    fab_batch = seed_photoanode_fabrication_batch(session)
    device = seed_device(
        session,
        experimental_condition_id=condition.id,
        photoanode_fabrication_batch_id=fab_batch.id,
        is_synthetic=True,
    )
    _add_pv_measurement(session, device.id)
    session.commit()

    df = build_ml_dataset(session)
    assert bool(df.iloc[0]["is_synthetic"]) is True
