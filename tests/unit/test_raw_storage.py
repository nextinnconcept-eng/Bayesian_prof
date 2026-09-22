"""Tests for raw instrument file storage (Chunk 2.1): register/resolve
round-trip, checksum-based tamper/corruption detection, and immutability
(refusing to overwrite an already-registered file).

tests/fixtures/sample_uvvis_spectrum.csv is dummy data used only to
exercise the storage mechanism, not real instrument output.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dssc_discovery.chemistry.raw_storage import (
    RawDataReference,
    RawFileStorageError,
    register_raw_file,
    resolve_raw_file,
)
from dssc_discovery.config import clear_config_cache
from dssc_discovery.domain.enums import MeasurementType

FIXTURE_PATH = Path(__file__).resolve().parent.parent / "fixtures" / "sample_uvvis_spectrum.csv"


@pytest.fixture(autouse=True)
def _isolated_data_dir(tmp_path, monkeypatch):
    # Route all storage into a temp directory -- never touch the real
    # project data/ folder from tests.
    monkeypatch.setenv("DSSC_DATA_DIR", str(tmp_path))
    clear_config_cache()
    yield
    clear_config_cache()


def _write_source_file(
    tmp_path: Path, name: str = "source.csv", content: str = "a,b\n1,2\n"
) -> Path:
    path = tmp_path / name
    path.write_text(content, encoding="utf-8")
    return path


def test_register_then_resolve_returns_unmodified_bytes(tmp_path):
    source = _write_source_file(tmp_path)
    reference = register_raw_file(source, MeasurementType.UV_VIS, dest_filename="a.csv")

    resolved = resolve_raw_file(reference.to_reference_string())

    assert resolved == source.read_bytes()


def test_register_real_dummy_spectrum_file_round_trips_losslessly(tmp_path):
    reference = register_raw_file(FIXTURE_PATH, MeasurementType.UV_VIS, dest_filename="uvvis.csv")
    resolved = resolve_raw_file(reference.to_reference_string())
    assert resolved == FIXTURE_PATH.read_bytes()


def test_register_stores_under_measurement_type_subdirectory(tmp_path):
    source = _write_source_file(tmp_path)
    reference = register_raw_file(source, MeasurementType.EIS, dest_filename="a.csv")
    assert reference.path.startswith("raw/eis/")


def test_original_source_file_is_untouched(tmp_path):
    source = _write_source_file(tmp_path, content="original content\n")
    original_bytes = source.read_bytes()

    register_raw_file(source, MeasurementType.UV_VIS, dest_filename="copy.csv")

    assert source.read_bytes() == original_bytes  # copy, not move


def test_tampering_with_stored_file_is_detected(tmp_path):
    source = _write_source_file(tmp_path)
    reference = register_raw_file(source, MeasurementType.UV_VIS, dest_filename="a.csv")

    from dssc_discovery.config import resolve_path

    stored_path = resolve_path(Path(tmp_path)) / reference.path
    stored_path.write_bytes(b"tampered content")

    with pytest.raises(RawFileStorageError, match="Checksum mismatch"):
        resolve_raw_file(reference.to_reference_string())


def test_resolve_missing_file_raises(tmp_path):
    reference = RawDataReference(path="raw/uv_vis/does_not_exist.csv", sha256="0" * 64)
    with pytest.raises(RawFileStorageError, match="missing"):
        resolve_raw_file(reference.to_reference_string())


def test_resolve_malformed_reference_string_raises():
    with pytest.raises(RawFileStorageError, match="Malformed"):
        resolve_raw_file("not valid json")


def test_register_refuses_to_overwrite_existing_registration(tmp_path):
    source = _write_source_file(tmp_path)
    register_raw_file(source, MeasurementType.UV_VIS, dest_filename="dup.csv")

    with pytest.raises(RawFileStorageError, match="already-registered"):
        register_raw_file(source, MeasurementType.UV_VIS, dest_filename="dup.csv")


def test_register_missing_source_raises(tmp_path):
    with pytest.raises(RawFileStorageError, match="does not exist"):
        register_raw_file(tmp_path / "nonexistent.csv", MeasurementType.UV_VIS)


def test_reference_round_trips_through_string_form():
    reference = RawDataReference(path="raw/uv_vis/x.csv", sha256="abc123")
    restored = RawDataReference.from_reference_string(reference.to_reference_string())
    assert restored == reference


def test_reference_string_fits_within_db_column_length(tmp_path):
    # RawSpectrumORM.raw_data_reference is String(500) (Chunk 1.4) --
    # a realistic path + sha256 must comfortably fit.
    source = _write_source_file(tmp_path)
    reference = register_raw_file(
        source, MeasurementType.UV_VIS, dest_filename="a_reasonably_long_filename_2026.csv"
    )
    assert len(reference.to_reference_string()) < 500
