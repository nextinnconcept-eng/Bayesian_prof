"""Integration test: a synthetic Gaussian UV-Vis spectrum registered via
chemistry.raw_storage (Chunk 2.1), processed end-to-end by
derive_uvvis_descriptors (Chunk 2.2), and the resulting DerivedDescriptor
objects persisted through the repository layer -- the full raw-file ->
descriptor -> database path.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pytest

from dssc_discovery.chemistry.raw_storage import register_raw_file
from dssc_discovery.chemistry.uvvis import PROCESSING_VERSION, derive_uvvis_descriptors
from dssc_discovery.config import clear_config_cache
from dssc_discovery.database.repositories import DerivedDescriptorRepository, RawSpectrumRepository
from dssc_discovery.domain.enums import MeasurementType, SpeciesCode
from dssc_discovery.domain.measurements import RawSpectrum
from tests.seed_helpers import seed_pigment_extract_batch, seed_plant_sample, seed_species

AMPLITUDE, MU, SIGMA = 0.85, 520.0, 15.0
ANALYTICAL_FWHM = 2 * np.sqrt(2 * np.log(2)) * SIGMA
ANALYTICAL_INTEGRAL = AMPLITUDE * SIGMA * np.sqrt(2 * np.pi)


def _write_gaussian_spectrum_csv(path):
    wavelength = np.arange(400.0, 650.5, 0.5)
    absorbance = AMPLITUDE * np.exp(-((wavelength - MU) ** 2) / (2 * SIGMA**2))
    lines = ["wavelength_nm,absorbance"]
    lines += [f"{w},{a}" for w, a in zip(wavelength, absorbance)]
    path.write_text("\n".join(lines), encoding="utf-8")


@pytest.fixture(autouse=True)
def _isolated_data_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("DSSC_DATA_DIR", str(tmp_path))
    clear_config_cache()
    yield
    clear_config_cache()


def _seed_uvvis_raw_spectrum(session, reference):
    seed_species(session, SpeciesCode.BETA_VULGARIS)
    sample = seed_plant_sample(session, SpeciesCode.BETA_VULGARIS)
    batch = seed_pigment_extract_batch(session, plant_sample_id=sample.id)
    return RawSpectrumRepository(session).add(
        RawSpectrum(
            measurement_type=MeasurementType.UV_VIS,
            pigment_extract_batch_id=batch.id,
            raw_data_reference=reference.to_reference_string(),
            acquisition_date=date(2026, 1, 15),
            operator="J. Doe",
            instrument="Shimadzu UV-1800",
        )
    )


def test_full_pipeline_register_derive_persist(session, tmp_path):
    source = tmp_path / "gaussian_uvvis.csv"
    _write_gaussian_spectrum_csv(source)
    reference = register_raw_file(source, MeasurementType.UV_VIS, dest_filename="spectrum.csv")
    raw_spectrum = _seed_uvvis_raw_spectrum(session, reference)
    session.commit()

    descriptors = derive_uvvis_descriptors(
        source_spectrum_id=raw_spectrum.id,
        raw_data_reference=reference.to_reference_string(),
        operator="J. Doe",
    )

    assert len(descriptors) == 3
    by_name = {d.descriptor_name: d for d in descriptors}
    assert by_name["lambda_max_nm"].value == pytest.approx(MU, abs=0.5)
    assert by_name["bandwidth_nm"].value == pytest.approx(ANALYTICAL_FWHM, abs=0.01)
    assert by_name["integrated_absorption"].value == pytest.approx(ANALYTICAL_INTEGRAL, rel=1e-3)

    for descriptor in descriptors:
        assert descriptor.source_spectrum_id == raw_spectrum.id
        assert descriptor.processing_version == PROCESSING_VERSION

    repo = DerivedDescriptorRepository(session)
    persisted = [repo.add(d) for d in descriptors]
    session.commit()

    refetched = [repo.get(d.id) for d in persisted]
    assert all(r is not None for r in refetched)
    refetched_by_name = {d.descriptor_name: d for d in refetched}
    assert refetched_by_name["lambda_max_nm"].value == pytest.approx(MU, abs=0.5)


def test_reproducible_given_same_spectrum_and_processing_version(tmp_path):
    source = tmp_path / "gaussian_uvvis.csv"
    _write_gaussian_spectrum_csv(source)
    reference = register_raw_file(source, MeasurementType.UV_VIS, dest_filename="spectrum.csv")

    from uuid import uuid4

    spectrum_id = uuid4()
    first = derive_uvvis_descriptors(spectrum_id, reference.to_reference_string(), operator="A")
    second = derive_uvvis_descriptors(spectrum_id, reference.to_reference_string(), operator="A")

    first_values = {d.descriptor_name: d.value for d in first}
    second_values = {d.descriptor_name: d.value for d in second}
    assert first_values == second_values
    assert {d.processing_version for d in first} == {PROCESSING_VERSION}
