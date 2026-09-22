"""Integration test: a synthetic EIS spectrum registered via
chemistry.raw_storage (Chunk 2.1), fit end-to-end by
derive_eis_measurement (Chunk 2.4), and the resulting
ElectrochemicalMeasurement persisted through the repository layer --
both for a converged fit and for a deliberately non-convergent one,
proving the failure path produces a real, persistable record (Chunk
1.3's missing_value_reason design), not a crash.
"""

from __future__ import annotations

import numpy as np
import pytest

from dssc_discovery.chemistry.eis import derive_eis_measurement
from dssc_discovery.chemistry.raw_storage import register_raw_file
from dssc_discovery.config import clear_config_cache
from dssc_discovery.database.repositories import (
    ElectrochemicalMeasurementRepository,
)
from dssc_discovery.domain.enums import EquivalentCircuitModel, MeasurementType, SpeciesCode
from tests.seed_helpers import seed_full_device_chain, seed_raw_spectrum_for_device

RS_TRUE, RCT_TRUE, CCE_TRUE, RREC_TRUE, CMU_TRUE = 15.0, 50.0, 2e-6, 300.0, 5e-5
TAU_TRUE_MS = RREC_TRUE * CMU_TRUE * 1000.0


def _write_eis_spectrum_csv(path, n_points=60):
    frequency = np.logspace(-1, 5, n_points)
    omega = 2 * np.pi * frequency
    z_ce = RCT_TRUE / (1 + 1j * omega * RCT_TRUE * CCE_TRUE)
    z_rec = RREC_TRUE / (1 + 1j * omega * RREC_TRUE * CMU_TRUE)
    z = RS_TRUE + z_ce + z_rec
    lines = ["frequency_hz,z_real_ohm,z_imag_ohm"]
    lines += [f"{f},{r},{i}" for f, r, i in zip(frequency, z.real, z.imag)]
    path.write_text("\n".join(lines), encoding="utf-8")


@pytest.fixture(autouse=True)
def _isolated_data_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("DSSC_DATA_DIR", str(tmp_path))
    clear_config_cache()
    yield
    clear_config_cache()


def _seed_eis_raw_spectrum(session, device_id, reference):
    return seed_raw_spectrum_for_device(
        session,
        device_id=device_id,
        measurement_type=MeasurementType.EIS,
        raw_data_reference=reference.to_reference_string(),
    )


def test_converged_fit_full_pipeline_register_derive_persist(session, tmp_path):
    device = seed_full_device_chain(session, species_code=SpeciesCode.BETA_VULGARIS)
    source = tmp_path / "eis_spectrum.csv"
    _write_eis_spectrum_csv(source)
    reference = register_raw_file(source, MeasurementType.EIS, dest_filename="spectrum.csv")
    raw_spectrum = _seed_eis_raw_spectrum(session, device.id, reference)
    session.commit()

    measurement = derive_eis_measurement(
        device_id=device.id,
        source_spectrum_id=raw_spectrum.id,
        raw_data_reference=reference.to_reference_string(),
        operator="J. Doe",
        instrument="Autolab PGSTAT",
    )

    assert measurement.equivalent_circuit_model is EquivalentCircuitModel.RANDLES
    assert measurement.series_resistance_ohm == pytest.approx(RS_TRUE, rel=1e-3)
    assert measurement.recombination_resistance_ohm == pytest.approx(RREC_TRUE, rel=1e-3)
    assert measurement.electron_lifetime_ms == pytest.approx(TAU_TRUE_MS, rel=1e-3)
    assert measurement.fit_r_squared == pytest.approx(1.0, abs=1e-6)
    assert measurement.missing_value_reason is None

    repo = ElectrochemicalMeasurementRepository(session)
    persisted = repo.add(measurement)
    session.commit()

    refetched = repo.get(persisted.id)
    assert refetched is not None
    assert refetched.series_resistance_ohm == pytest.approx(RS_TRUE, rel=1e-3)


def test_non_convergent_fit_still_produces_a_persistable_record(session, tmp_path):
    device = seed_full_device_chain(session, species_code=SpeciesCode.BETA_VULGARIS)
    source = tmp_path / "eis_spectrum.csv"
    _write_eis_spectrum_csv(source)
    reference = register_raw_file(source, MeasurementType.EIS, dest_filename="spectrum.csv")
    raw_spectrum = _seed_eis_raw_spectrum(session, device.id, reference)
    session.commit()

    # force non-convergence via an exhausted iteration budget (empirically
    # confirmed reliable in Chunk 2.4's unit tests) rather than crashing
    measurement = derive_eis_measurement(
        device_id=device.id,
        source_spectrum_id=raw_spectrum.id,
        raw_data_reference=reference.to_reference_string(),
        operator="J. Doe",
        instrument="Autolab PGSTAT",
        max_nfev=1,
    )

    assert measurement.series_resistance_ohm is None
    assert measurement.charge_transfer_resistance_ohm is None
    assert measurement.recombination_resistance_ohm is None
    assert measurement.electron_lifetime_ms is None
    assert measurement.fit_r_squared is None
    assert measurement.missing_value_reason is not None
    assert "did not converge" in measurement.notes.lower()

    # the whole point of Chunk 1.3's Optional+missing_value_reason design:
    # a non-convergent fit still persists as a real, queryable attempt.
    repo = ElectrochemicalMeasurementRepository(session)
    persisted = repo.add(measurement)
    session.commit()

    refetched = repo.get(persisted.id)
    assert refetched is not None
    assert refetched.series_resistance_ohm is None
    assert refetched.missing_value_reason is not None


def test_reproducible_given_same_spectrum(session, tmp_path):
    device = seed_full_device_chain(session, species_code=SpeciesCode.BETA_VULGARIS)
    source = tmp_path / "eis_spectrum.csv"
    _write_eis_spectrum_csv(source)
    reference = register_raw_file(source, MeasurementType.EIS, dest_filename="spectrum.csv")
    raw_spectrum = _seed_eis_raw_spectrum(session, device.id, reference)
    session.commit()

    kwargs = dict(
        device_id=device.id,
        source_spectrum_id=raw_spectrum.id,
        raw_data_reference=reference.to_reference_string(),
        operator="J. Doe",
        instrument="Autolab PGSTAT",
    )
    first = derive_eis_measurement(**kwargs)
    second = derive_eis_measurement(**kwargs)

    assert first.series_resistance_ohm == second.series_resistance_ohm
    assert first.electron_lifetime_ms == second.electron_lifetime_ms
    assert first.fit_r_squared == second.fit_r_squared
