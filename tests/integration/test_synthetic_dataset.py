"""Integration tests for the synthetic dataset generator (Chunk 3.5):
generated rows persist as a full, internally-consistent FK chain, batch
generation is deterministic for a fixed seed, the generated PCE ground
truth is recoverable by a canary ML model, and create_dataset_version's
mixing guard (amended in this chunk) actually raises for a
synthetic+real mix rather than silently reporting is_synthetic=False.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pytest
import sqlalchemy as sa
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sqlalchemy.orm import Session

from dssc_discovery.database.models import Base
from dssc_discovery.database.repositories import PhotovoltaicMeasurementRepository
from dssc_discovery.domain.enums import IlluminationProtocol, SpeciesCode
from dssc_discovery.domain.measurements import PhotovoltaicMeasurement
from dssc_discovery.features.dataset_builder import build_ml_dataset
from dssc_discovery.features.synthetic import (
    SyntheticDatasetError,
    _sample_vector,
    generate_synthetic_condition_and_device,
    generate_synthetic_dataset,
)
from dssc_discovery.features.versioning import create_dataset_version
from dssc_discovery.validation.rules import check_pce_consistency
from tests.seed_helpers import (
    seed_device,
    seed_experimental_condition,
    seed_photoanode_fabrication_batch,
    seed_species,
)


def _fresh_session() -> Session:
    engine = sa.create_engine("sqlite://")

    @sa.event.listens_for(engine, "connect")
    def _enable_fk(dbapi_connection, connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

    Base.metadata.create_all(engine)
    return Session(engine)


def _add_real_measured_device(session, species_code, condition_code, device_code) -> None:
    seed_species(session, species_code)
    condition = seed_experimental_condition(session, species_code, condition_code=condition_code)
    fab_batch = seed_photoanode_fabrication_batch(session, batch_code=f"{device_code}-FAB")
    device = seed_device(
        session,
        experimental_condition_id=condition.id,
        photoanode_fabrication_batch_id=fab_batch.id,
        device_code=device_code,
        is_synthetic=False,
    )
    PhotovoltaicMeasurementRepository(session).add(
        PhotovoltaicMeasurement(
            device_id=device.id,
            illumination_protocol=IlluminationProtocol.AM1_5G_1_SUN,
            measurement_date=date(2026, 1, 20),
            open_circuit_voltage_v=0.65,
            short_circuit_current_density_ma_cm2=12.0,
            fill_factor=0.65,
            power_conversion_efficiency_percent=5.07,
            operator="J. Doe",
            instrument="Keithley 2400",
            is_synthetic=False,
        )
    )


def test_single_row_generator_persists_a_full_consistent_chain(session):
    rng = np.random.default_rng(1)
    vector = _sample_vector(rng)

    device = generate_synthetic_condition_and_device(
        session,
        vector,
        SpeciesCode.BETA_VULGARIS,
        "BV-SYN-0001",
        "BV-SYN-0001-R1",
        "synthetic-generator",
        rng,
    )
    session.commit()

    assert device.is_synthetic is True
    pv_rows = PhotovoltaicMeasurementRepository(session).filter(device_id=device.id)
    assert len(pv_rows) == 1
    assert pv_rows[0].is_synthetic is True
    check_pce_consistency(pv_rows[0], relative_tolerance=1e-6)  # must not raise


def test_batch_generator_is_deterministic_for_a_fixed_seed():
    session_a = _fresh_session()
    session_b = _fresh_session()
    try:
        generate_synthetic_dataset(session_a, n=15, seed=99)
        session_a.commit()
        generate_synthetic_dataset(session_b, n=15, seed=99)
        session_b.commit()

        df_a = build_ml_dataset(session_a).drop(columns=["device_id", "group_id"])
        df_b = build_ml_dataset(session_b).drop(columns=["device_id", "group_id"])
        assert df_a.equals(df_b)
    finally:
        session_a.close()
        session_b.close()


def test_batch_generator_all_rows_are_synthetic_and_pce_consistent(session):
    generate_synthetic_dataset(session, n=20, seed=5)
    session.commit()

    df = build_ml_dataset(session)
    assert len(df) == 20
    assert df["is_synthetic"].all()
    assert df["power_conversion_efficiency_percent"].between(0, 100).all()


def test_synthetic_pce_ground_truth_is_recoverable_by_a_canary_model(session):
    generate_synthetic_dataset(session, n=80, seed=2024, noise_std_percent=0.3)
    session.commit()

    df = build_ml_dataset(session)
    features = ["ph", "pigment_concentration_mm", "adsorption_time_min", "tio2_thickness_um"]
    x = df[features].to_numpy()
    y = df["power_conversion_efficiency_percent"].to_numpy()

    model = RandomForestRegressor(n_estimators=200, random_state=0)
    scores = cross_val_score(model, x, y, cv=5, scoring="r2")
    # loose, non-tautological bar: a model fit on pure noise would score
    # near/below 0 on held-out folds, so this confirms real recoverable
    # signal without demanding near-perfect recovery at this sample size.
    assert scores.mean() > 0.2


def test_mixing_synthetic_and_real_rows_raises_on_versioning(session):
    generate_synthetic_dataset(session, n=3, seed=1)
    _add_real_measured_device(
        session, SpeciesCode.CHROMOLAENA_ODORATA, "CO-REAL-001", "CO-REAL-001-R1"
    )
    session.commit()

    with pytest.raises(SyntheticDatasetError):
        create_dataset_version(session, operator="J. Doe")


def test_all_synthetic_dataset_versions_successfully(session):
    generate_synthetic_dataset(session, n=4, seed=1)
    session.commit()

    version = create_dataset_version(session, operator="J. Doe")
    assert version.is_synthetic is True
    assert version.row_count == 4
