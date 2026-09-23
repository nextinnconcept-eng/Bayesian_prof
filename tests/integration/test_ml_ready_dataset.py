"""M3 exit-gate integration test (Chunk 3.6): the full chain --
DB fixture -> dataset assembly (3.2) -> dataset versioning (3.3) ->
leakage-safe preprocessing pipeline (3.4) -- runs clean end-to-end, for
both a real-like heterogeneous dataset and the synthetic generator's
output (3.5), and versioning/preprocessing compose correctly on the
dataset build_ml_dataset() actually produces (not just hand-built toy
DataFrames, which is all 3.4's own tests use).

build_ml_dataset()'s own join correctness (row selection, group_id,
missing-characterization-as-NaN, reproducibility) is already thoroughly
covered by tests/integration/test_dataset_builder.py (Chunk 3.2) and is
deliberately not re-proven here -- this file only tests the
COMPOSITION of 3.1-3.5 that no earlier chunk's tests exercise together.
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
    SpeciesRepository,
)
from dssc_discovery.domain.enums import (
    EquivalentCircuitModel,
    IlluminationProtocol,
    MeasurementType,
    SolventType,
    SpeciesCode,
    SurfaceTreatment,
)
from dssc_discovery.domain.experimental_condition import ExperimentalConditionVector
from dssc_discovery.domain.measurements import (
    DerivedDescriptor,
    ElectrochemicalMeasurement,
    RawSpectrum,
)
from dssc_discovery.domain.measurements import PhotovoltaicMeasurement as PVMeasurement
from dssc_discovery.features.dataset_builder import build_ml_dataset
from dssc_discovery.features.registry import FEATURE_REGISTRY, design_features
from dssc_discovery.features.synthetic import generate_synthetic_dataset
from dssc_discovery.features.versioning import create_dataset_version
from dssc_discovery.preprocessing.pipelines import build_preprocessing_pipeline
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
    return PhotovoltaicMeasurementRepository(session).add(PVMeasurement(**kwargs))


def _vector_with_ph(ph: float) -> ExperimentalConditionVector:
    return ExperimentalConditionVector(
        solvent=SolventType.ETHANOL,
        solvent_ratio=0.7,
        ph=ph,
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


def _seed_condition_with_device(session, species_code, condition_code, ph):
    # get-or-create: seed_species() has no such guard of its own, and
    # this helper is deliberately called more than once per species
    # across a single test to build up a multi-condition dataset.
    if SpeciesRepository(session).get_one(code=species_code) is None:
        seed_species(session, species_code)
    condition = seed_experimental_condition(
        session,
        species_code,
        condition_code=condition_code,
        target_parameters=_vector_with_ph(ph),
    )
    fab_batch = seed_photoanode_fabrication_batch(session, batch_code=f"{condition_code}-FAB")
    device = seed_device(
        session,
        experimental_condition_id=condition.id,
        photoanode_fabrication_batch_id=fab_batch.id,
        device_code=f"{condition_code}-R1",
    )
    return condition, device


def _attach_full_characterization(session, condition, device):
    """SensitizationRun + PigmentExtractBatch + UV-Vis descriptors + EIS
    -- exercises the characterization-feature columns that a device with
    no such records leaves as NaN (verified separately in Chunk 3.2's
    own tests)."""
    sample = seed_plant_sample(session, condition.species_code, _suffix=condition.condition_code)
    pigment_batch = seed_pigment_extract_batch(
        session,
        plant_sample_id=sample.id,
        batch_code=f"{condition.condition_code}-EXT",
        extraction_yield_percent=40.0,
    )
    seed_sensitization_run(session, device_id=device.id, pigment_extract_batch_id=pigment_batch.id)

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

    eis_spectrum = RawSpectrumRepository(session).add(
        RawSpectrum(
            measurement_type=MeasurementType.EIS,
            device_id=device.id,
            raw_data_reference='{"path": "x", "sha256": "y"}',
            acquisition_date=date(2026, 1, 20),
            operator="J. Doe",
            instrument="Autolab PGSTAT",
        )
    )
    ElectrochemicalMeasurementRepository(session).add(
        ElectrochemicalMeasurement(
            device_id=device.id,
            raw_spectrum_id=eis_spectrum.id,
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


def test_real_like_dataset_flows_clean_through_assembly_versioning_and_preprocessing(session):
    _condition_a, device_a = _seed_condition_with_device(
        session, SpeciesCode.BETA_VULGARIS, "BV-COND-001", ph=5.0
    )
    _attach_full_characterization(session, _condition_a, device_a)
    _add_pv_measurement(session, device_a.id, power_conversion_efficiency_percent=5.07)

    _condition_b, device_b = _seed_condition_with_device(
        session, SpeciesCode.BETA_VULGARIS, "BV-COND-002", ph=6.5
    )
    _add_pv_measurement(session, device_b.id, power_conversion_efficiency_percent=3.2)

    _condition_c, device_c = _seed_condition_with_device(
        session, SpeciesCode.CHROMOLAENA_ODORATA, "CO-COND-001", ph=4.2
    )
    _add_pv_measurement(session, device_c.id, power_conversion_efficiency_percent=6.1)

    _condition_d, device_d = _seed_condition_with_device(
        session, SpeciesCode.CHROMOLAENA_ODORATA, "CO-COND-002", ph=7.8
    )
    _add_pv_measurement(session, device_d.id, power_conversion_efficiency_percent=2.4)
    session.commit()

    # --- dataset assembly (3.2) ---
    dataframe = build_ml_dataset(session)
    assert len(dataframe) == 4
    assert not np.isnan(
        dataframe.loc[dataframe["device_code"] == "BV-COND-001-R1", "lambda_max_nm"]
    ).any()
    assert np.isnan(
        dataframe.loc[dataframe["device_code"] == "BV-COND-002-R1", "lambda_max_nm"]
    ).all()

    # --- versioning (3.3): valid, and deterministic for unchanged state ---
    version = create_dataset_version(session, operator="J. Doe")
    assert version.row_count == 4
    assert version.is_synthetic is False
    version_again = create_dataset_version(session, operator="J. Doe")
    assert version_again.dataset_hash == version.dataset_hash

    # --- preprocessing (3.4): fit over every registered feature, real data ---
    feature_columns = [spec.name for spec in FEATURE_REGISTRY]
    pipeline = build_preprocessing_pipeline(list(FEATURE_REGISTRY))
    transformed = pipeline.fit_transform(dataframe[feature_columns])

    assert transformed.shape[0] == 4
    assert not np.isnan(
        transformed
    ).any()  # imputation must fill every NaN, including EIS/UV-Vis gaps


def test_grouped_train_test_split_is_leakage_safe_on_the_real_assembled_dataset(session):
    # Two species/conditions with visibly different pH -- if test-fold
    # data leaked into the fit, the scaler's train-only statistics
    # would be pulled toward the test values instead of matching the
    # train-only mean/variance exactly.
    _condition_train, device_train = _seed_condition_with_device(
        session, SpeciesCode.BETA_VULGARIS, "BV-COND-001", ph=5.0
    )
    _add_pv_measurement(session, device_train.id)

    _condition_test, device_test = _seed_condition_with_device(
        session, SpeciesCode.CHROMOLAENA_ODORATA, "CO-COND-001", ph=13.5
    )
    _add_pv_measurement(session, device_test.id)
    session.commit()

    dataframe = build_ml_dataset(session)
    train_df = dataframe[dataframe["group_id"] == str(_condition_train.id)]
    test_df = dataframe[dataframe["group_id"] == str(_condition_test.id)]
    assert len(train_df) == 1
    assert len(test_df) == 1

    feature_columns = [spec.name for spec in design_features()]
    pipeline = build_preprocessing_pipeline(list(design_features()))
    pipeline.fit(train_df[feature_columns])

    scaler = pipeline.named_transformers_["continuous"].named_steps["scale"]
    ph_index = [f.name for f in design_features() if f.dtype == "continuous"].index("ph")
    assert scaler.mean_[ph_index] == pytest.approx(5.0)  # train-only, not blended with test's 13.5

    transformed_test = pipeline.transform(test_df[feature_columns])
    assert transformed_test.shape[0] == 1
    assert not np.isnan(transformed_test).any()


def test_synthetic_dataset_also_completes_the_full_m3_chain(session):
    generate_synthetic_dataset(session, n=10, seed=123)
    session.commit()

    dataframe = build_ml_dataset(session)
    assert len(dataframe) == 10

    version = create_dataset_version(session, operator="J. Doe")
    assert version.is_synthetic is True
    assert version.row_count == 10

    feature_columns = [spec.name for spec in design_features()]
    pipeline = build_preprocessing_pipeline(list(design_features()))
    transformed = pipeline.fit_transform(dataframe[feature_columns])

    assert transformed.shape[0] == 10
    assert not np.isnan(transformed).any()
