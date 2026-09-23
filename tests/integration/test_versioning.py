"""Tests for dataset version snapshotting (Chunk 3.3): identical DB state
produces an identical hash, changed DB state produces a different one,
and the resulting MLDatasetVersion is a real, persistable, immutable
record any future ModelRun could cite.

Git-hash capture is mocked throughout (not the real repo's live commit
hash) -- tests must not depend on which commit happens to be checked out
when they run.
"""

from __future__ import annotations

import json
from datetime import date

import pytest

from dssc_discovery.database.repositories import (
    MLDatasetVersionRepository,
    PhotovoltaicMeasurementRepository,
)
from dssc_discovery.domain.enums import IlluminationProtocol, SpeciesCode
from dssc_discovery.domain.measurements import PhotovoltaicMeasurement
from dssc_discovery.features.synthetic import SyntheticDatasetError
from dssc_discovery.features.versioning import compute_dataset_hash, create_dataset_version
from tests.seed_helpers import (
    seed_device,
    seed_experimental_condition,
    seed_photoanode_fabrication_batch,
    seed_species,
)


@pytest.fixture(autouse=True)
def _mock_git(monkeypatch):
    monkeypatch.setattr("dssc_discovery.features.versioning.get_git_commit_hash", lambda: "abc123")
    monkeypatch.setattr(
        "dssc_discovery.features.versioning.is_git_working_tree_dirty", lambda: False
    )


def _add_measured_device(
    session, species_code, condition_code, device_code, is_synthetic=False, **pv_overrides
):
    seed_species(session, species_code)
    condition = seed_experimental_condition(session, species_code, condition_code=condition_code)
    fab_batch = seed_photoanode_fabrication_batch(session, batch_code=f"{device_code}-FAB")
    device = seed_device(
        session,
        experimental_condition_id=condition.id,
        photoanode_fabrication_batch_id=fab_batch.id,
        device_code=device_code,
        is_synthetic=is_synthetic,
    )
    kwargs = dict(
        device_id=device.id,
        illumination_protocol=IlluminationProtocol.AM1_5G_1_SUN,
        measurement_date=date(2026, 1, 20),
        open_circuit_voltage_v=0.65,
        short_circuit_current_density_ma_cm2=12.0,
        fill_factor=0.65,
        power_conversion_efficiency_percent=5.07,
        operator="J. Doe",
        instrument="Keithley 2400",
    )
    kwargs.update(pv_overrides)
    PhotovoltaicMeasurementRepository(session).add(PhotovoltaicMeasurement(**kwargs))
    return device


# --- compute_dataset_hash: pure function --------------------------------------


def test_hash_is_deterministic_for_identical_inputs():
    h1 = compute_dataset_hash(["a", "b", "c"], "build_ml_dataset", "abc123")
    h2 = compute_dataset_hash(["a", "b", "c"], "build_ml_dataset", "abc123")
    assert h1 == h2


def test_hash_is_independent_of_row_id_order():
    h1 = compute_dataset_hash(["a", "b", "c"], "build_ml_dataset", "abc123")
    h2 = compute_dataset_hash(["c", "a", "b"], "build_ml_dataset", "abc123")
    assert h1 == h2


def test_hash_changes_with_different_row_ids():
    h1 = compute_dataset_hash(["a", "b", "c"], "build_ml_dataset", "abc123")
    h2 = compute_dataset_hash(["a", "b", "d"], "build_ml_dataset", "abc123")
    assert h1 != h2


def test_hash_changes_with_different_code_version():
    h1 = compute_dataset_hash(["a", "b", "c"], "build_ml_dataset", "abc123")
    h2 = compute_dataset_hash(["a", "b", "c"], "build_ml_dataset", "def456")
    assert h1 != h2


def test_hash_changes_with_different_query_definition():
    h1 = compute_dataset_hash(["a", "b", "c"], "build_ml_dataset", "abc123")
    h2 = compute_dataset_hash(["a", "b", "c"], "build_ml_dataset_v2", "abc123")
    assert h1 != h2


# --- create_dataset_version: integration --------------------------------------


def test_identical_db_state_produces_identical_hash(session):
    _add_measured_device(session, SpeciesCode.BETA_VULGARIS, "BV-COND-001", "BV-DEV-001")
    session.commit()

    v1 = create_dataset_version(session, operator="J. Doe")
    v2 = create_dataset_version(session, operator="J. Doe")

    assert v1.dataset_hash == v2.dataset_hash
    assert v1.row_count == v2.row_count == 1


def test_changed_db_state_produces_different_hash(session):
    _add_measured_device(session, SpeciesCode.BETA_VULGARIS, "BV-COND-001", "BV-DEV-001")
    session.commit()
    v1 = create_dataset_version(session, operator="J. Doe")

    _add_measured_device(session, SpeciesCode.CHROMOLAENA_ODORATA, "CO-COND-001", "CO-DEV-001")
    session.commit()
    v2 = create_dataset_version(session, operator="J. Doe")

    assert v1.dataset_hash != v2.dataset_hash
    assert v2.row_count == 2


def test_row_ids_json_matches_the_actual_dataset_row_ids(session):
    device = _add_measured_device(session, SpeciesCode.BETA_VULGARIS, "BV-COND-001", "BV-DEV-001")
    session.commit()

    version = create_dataset_version(session, operator="J. Doe")
    row_ids = json.loads(version.row_ids_json)

    assert row_ids == [str(device.id)]


def test_feature_columns_json_is_valid_and_nonempty(session):
    _add_measured_device(session, SpeciesCode.BETA_VULGARIS, "BV-COND-001", "BV-DEV-001")
    session.commit()

    version = create_dataset_version(session, operator="J. Doe")
    columns = json.loads(version.feature_columns_json)

    assert "ph" in columns
    assert "power_conversion_efficiency_percent" in columns


def test_empty_database_still_produces_a_valid_version(session):
    version = create_dataset_version(session, operator="J. Doe")
    assert version.row_count == 0
    assert json.loads(version.row_ids_json) == []
    assert version.is_synthetic is False


def test_is_synthetic_true_when_every_row_is_synthetic(session):
    _add_measured_device(
        session, SpeciesCode.BETA_VULGARIS, "BV-COND-001", "BV-DEV-001", is_synthetic=True
    )
    session.commit()

    version = create_dataset_version(session, operator="J. Doe")
    assert version.is_synthetic is True


def test_mixed_synthetic_and_real_rows_raises_synthetic_dataset_error(session):
    _add_measured_device(
        session, SpeciesCode.BETA_VULGARIS, "BV-COND-001", "BV-DEV-001", is_synthetic=True
    )
    _add_measured_device(
        session, SpeciesCode.CHROMOLAENA_ODORATA, "CO-COND-001", "CO-DEV-001", is_synthetic=False
    )
    session.commit()

    with pytest.raises(SyntheticDatasetError):
        create_dataset_version(session, operator="J. Doe")


def test_version_round_trips_through_the_repository(session):
    _add_measured_device(session, SpeciesCode.BETA_VULGARIS, "BV-COND-001", "BV-DEV-001")
    session.commit()

    version = create_dataset_version(session, operator="J. Doe")
    session.commit()

    refetched = MLDatasetVersionRepository(session).get(version.id)
    assert refetched is not None
    assert refetched.dataset_hash == version.dataset_hash
    assert refetched.row_count == version.row_count


def test_code_version_and_dirty_flag_are_captured(session):
    _add_measured_device(session, SpeciesCode.BETA_VULGARIS, "BV-COND-001", "BV-DEV-001")
    session.commit()

    version = create_dataset_version(session, operator="J. Doe")
    assert version.code_version == "abc123"  # from the mocked get_git_commit_hash
    assert version.is_dirty is False  # from the mocked is_git_working_tree_dirty
