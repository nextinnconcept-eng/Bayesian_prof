"""Integration test: a raw file registered via chemistry.raw_storage,
referenced from a real RawSpectrum persisted through the repository
layer, and resolved back losslessly after a round-trip through the
database -- the full path Chunk 2.1's DoD describes.
"""

from __future__ import annotations

from datetime import date
from pathlib import Path

from dssc_discovery.chemistry.raw_storage import register_raw_file, resolve_raw_file
from dssc_discovery.config import clear_config_cache
from dssc_discovery.database.repositories import RawSpectrumRepository
from dssc_discovery.domain.enums import MeasurementType, SpeciesCode
from dssc_discovery.domain.measurements import RawSpectrum
from tests.seed_helpers import seed_pigment_extract_batch, seed_plant_sample, seed_species

FIXTURE_PATH = Path(__file__).resolve().parent.parent / "fixtures" / "sample_uvvis_spectrum.csv"


def test_registered_file_referenced_by_persisted_raw_spectrum_resolves_losslessly(
    session, tmp_path, monkeypatch
):
    monkeypatch.setenv("DSSC_DATA_DIR", str(tmp_path))
    clear_config_cache()

    reference = register_raw_file(FIXTURE_PATH, MeasurementType.UV_VIS, dest_filename="uvvis.csv")

    seed_species(session, SpeciesCode.BETA_VULGARIS)
    sample = seed_plant_sample(session, SpeciesCode.BETA_VULGARIS)
    batch = seed_pigment_extract_batch(session, plant_sample_id=sample.id)

    spectrum = RawSpectrumRepository(session).add(
        RawSpectrum(
            measurement_type=MeasurementType.UV_VIS,
            pigment_extract_batch_id=batch.id,
            raw_data_reference=reference.to_reference_string(),
            acquisition_date=date(2026, 1, 15),
            operator="J. Doe",
            instrument="Shimadzu UV-1800",
        )
    )
    session.commit()

    # simulate a fresh read, as if in a later process/session
    refetched = RawSpectrumRepository(session).get(spectrum.id)
    resolved_bytes = resolve_raw_file(refetched.raw_data_reference)

    assert resolved_bytes == FIXTURE_PATH.read_bytes()

    clear_config_cache()
