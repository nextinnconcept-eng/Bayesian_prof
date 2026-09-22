"""Integration test: a synthetic multi-peak FTIR spectrum registered via
chemistry.raw_storage (Chunk 2.1), processed end-to-end by
derive_ftir_descriptors (Chunk 2.3), and the resulting position/intensity
DerivedDescriptor rows persisted through the repository layer.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pytest

from dssc_discovery.chemistry.ftir import PROCESSING_VERSION, derive_ftir_descriptors
from dssc_discovery.chemistry.raw_storage import register_raw_file
from dssc_discovery.config import clear_config_cache
from dssc_discovery.database.repositories import DerivedDescriptorRepository, RawSpectrumRepository
from dssc_discovery.domain.enums import MeasurementType, SpeciesCode
from dssc_discovery.domain.measurements import RawSpectrum
from tests.seed_helpers import seed_pigment_extract_batch, seed_plant_sample, seed_species

GROUND_TRUTH_PEAKS = [(1050.0, 0.30, 15.0), (1715.0, 0.60, 12.0), (3300.0, 0.45, 25.0)]


def _write_ftir_spectrum_csv(path):
    wavenumber = np.arange(800.0, 3600.5, 1.0)
    absorbance = np.zeros_like(wavenumber)
    for position, amplitude, sigma in GROUND_TRUTH_PEAKS:
        absorbance += amplitude * np.exp(-((wavenumber - position) ** 2) / (2 * sigma**2))
    lines = ["wavenumber_cm1,absorbance"]
    lines += [f"{w},{a}" for w, a in zip(wavenumber, absorbance)]
    path.write_text("\n".join(lines), encoding="utf-8")


@pytest.fixture(autouse=True)
def _isolated_data_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("DSSC_DATA_DIR", str(tmp_path))
    clear_config_cache()
    yield
    clear_config_cache()


def _seed_ftir_raw_spectrum(session, reference):
    seed_species(session, SpeciesCode.BETA_VULGARIS)
    sample = seed_plant_sample(session, SpeciesCode.BETA_VULGARIS)
    batch = seed_pigment_extract_batch(session, plant_sample_id=sample.id)
    return RawSpectrumRepository(session).add(
        RawSpectrum(
            measurement_type=MeasurementType.FTIR,
            pigment_extract_batch_id=batch.id,
            raw_data_reference=reference.to_reference_string(),
            acquisition_date=date(2026, 1, 15),
            operator="J. Doe",
            instrument="Nicolet iS10",
        )
    )


def test_full_pipeline_register_derive_persist(session, tmp_path):
    source = tmp_path / "ftir_spectrum.csv"
    _write_ftir_spectrum_csv(source)
    reference = register_raw_file(source, MeasurementType.FTIR, dest_filename="spectrum.csv")
    raw_spectrum = _seed_ftir_raw_spectrum(session, reference)
    session.commit()

    descriptors = derive_ftir_descriptors(
        source_spectrum_id=raw_spectrum.id,
        raw_data_reference=reference.to_reference_string(),
        operator="J. Doe",
    )

    # 3 ground-truth peaks -> 6 rows (position + intensity each)
    assert len(descriptors) == 6
    for descriptor in descriptors:
        assert descriptor.source_spectrum_id == raw_spectrum.id
        assert descriptor.processing_version == PROCESSING_VERSION

    by_name = {d.descriptor_name: d.value for d in descriptors}
    assert by_name["ftir_peak_1_position"] == pytest.approx(1050.0, abs=1.0)
    assert by_name["ftir_peak_1_intensity"] == pytest.approx(0.30, abs=1e-6)
    assert by_name["ftir_peak_2_position"] == pytest.approx(1715.0, abs=1.0)
    assert by_name["ftir_peak_3_position"] == pytest.approx(3300.0, abs=1.0)

    repo = DerivedDescriptorRepository(session)
    persisted = [repo.add(d) for d in descriptors]
    session.commit()

    refetched = [repo.get(d.id) for d in persisted]
    assert all(r is not None for r in refetched)


def test_reproducible_given_same_spectrum_and_processing_version(tmp_path):
    from uuid import uuid4

    source = tmp_path / "ftir_spectrum.csv"
    _write_ftir_spectrum_csv(source)
    reference = register_raw_file(source, MeasurementType.FTIR, dest_filename="spectrum.csv")

    spectrum_id = uuid4()
    first = derive_ftir_descriptors(spectrum_id, reference.to_reference_string(), operator="A")
    second = derive_ftir_descriptors(spectrum_id, reference.to_reference_string(), operator="A")

    first_values = {d.descriptor_name: d.value for d in first}
    second_values = {d.descriptor_name: d.value for d in second}
    assert first_values == second_values
    assert {d.processing_version for d in first} == {PROCESSING_VERSION}
