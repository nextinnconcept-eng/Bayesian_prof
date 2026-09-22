"""M2 exit gate: one coherent end-to-end scientific scenario tying all
four characterization modalities together -- not re-verifying each
modality's own numerics (that's Chunks 2.2-2.5's job), but proving they
work TOGETHER with a fully traceable provenance chain, which is the
actual point of M2 as a milestone.

Scenario: a pigment batch (UV-Vis + FTIR characterized) sensitizes a
device (EIS + J-V characterized) -- the same shape a real DSSC
characterization campaign takes. Every raw file is registered with a
checksum (Chunk 2.1); every derived result traces back to its source
RawSpectrum; every RawSpectrum traces back to the same underlying
species/condition. This is the point full raw-instrument-data ingestion
becomes trustworthy for real data.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pytest

from dssc_discovery.chemistry.eis import derive_eis_measurement
from dssc_discovery.chemistry.ftir import derive_ftir_descriptors
from dssc_discovery.chemistry.jv import derive_jv_measurement
from dssc_discovery.chemistry.raw_storage import register_raw_file, resolve_raw_file
from dssc_discovery.chemistry.uvvis import derive_uvvis_descriptors
from dssc_discovery.config import clear_config_cache
from dssc_discovery.database.audit import AppendOnlyViolationError
from dssc_discovery.database.models import RawSpectrumORM
from dssc_discovery.database.repositories import (
    DerivedDescriptorRepository,
    ElectrochemicalMeasurementRepository,
    PhotovoltaicMeasurementRepository,
    RawSpectrumRepository,
)
from dssc_discovery.domain.enums import IlluminationProtocol, MeasurementType, SpeciesCode
from dssc_discovery.domain.measurements import RawSpectrum
from tests.seed_helpers import (
    seed_full_device_chain,
    seed_pigment_extract_batch,
    seed_plant_sample,
    seed_sensitization_run,
)


@pytest.fixture(autouse=True)
def _isolated_data_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("DSSC_DATA_DIR", str(tmp_path))
    clear_config_cache()
    yield
    clear_config_cache()


def _write_uvvis_csv(path):
    wavelength = np.arange(400.0, 650.5, 0.5)
    absorbance = 0.85 * np.exp(-((wavelength - 520.0) ** 2) / (2 * 15.0**2))
    lines = ["wavelength_nm,absorbance"] + [f"{w},{a}" for w, a in zip(wavelength, absorbance)]
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_ftir_csv(path):
    wavenumber = np.arange(800.0, 3600.5, 1.0)
    peaks = [(1050.0, 0.30, 15.0), (1715.0, 0.60, 12.0), (3300.0, 0.45, 25.0)]
    absorbance = np.zeros_like(wavenumber)
    for position, amplitude, sigma in peaks:
        absorbance += amplitude * np.exp(-((wavenumber - position) ** 2) / (2 * sigma**2))
    lines = ["wavenumber_cm1,absorbance"] + [f"{w},{a}" for w, a in zip(wavenumber, absorbance)]
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_eis_csv(path):
    rs, rct, cce, rrec, cmu = 15.0, 50.0, 2e-6, 300.0, 5e-5
    frequency = np.logspace(-1, 5, 60)
    omega = 2 * np.pi * frequency
    z_ce = rct / (1 + 1j * omega * rct * cce)
    z_rec = rrec / (1 + 1j * omega * rrec * cmu)
    z = rs + z_ce + z_rec
    lines = ["frequency_hz,z_real_ohm,z_imag_ohm"]
    lines += [f"{f},{r},{i}" for f, r, i in zip(frequency, z.real, z.imag)]
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_jv_csv(path):
    jsc, voc, n_ideality, vt = 12.0, 0.68, 1.5, 0.02585
    j0 = jsc / (np.exp(voc / (n_ideality * vt)) - 1)
    voltage = np.arange(0.0, 0.75 + 0.005, 0.005)
    current = jsc - j0 * (np.exp(voltage / (n_ideality * vt)) - 1)
    lines = ["voltage_v,current_density_ma_cm2"]
    lines += [f"{v},{j}" for v, j in zip(voltage, current)]
    path.write_text("\n".join(lines), encoding="utf-8")


def test_full_characterization_pipeline_all_four_modalities(session, tmp_path):
    # --- one coherent scientific scenario: pigment batch sensitizes a device ---
    # seed_full_device_chain() creates the species row as a side effect (it
    # has no get-or-create check), so it must run first; seed_plant_sample()
    # then reuses that same already-existing species for its own FK.
    device = seed_full_device_chain(session, species_code=SpeciesCode.BETA_VULGARIS)
    sample = seed_plant_sample(session, SpeciesCode.BETA_VULGARIS)
    pigment_batch = seed_pigment_extract_batch(session, plant_sample_id=sample.id)
    seed_sensitization_run(session, device_id=device.id, pigment_extract_batch_id=pigment_batch.id)
    session.commit()

    # --- register all four raw files (Chunk 2.1: copy + checksum) --------------
    uvvis_source = tmp_path / "uvvis.csv"
    ftir_source = tmp_path / "ftir.csv"
    eis_source = tmp_path / "eis.csv"
    jv_source = tmp_path / "jv.csv"
    _write_uvvis_csv(uvvis_source)
    _write_ftir_csv(ftir_source)
    _write_eis_csv(eis_source)
    _write_jv_csv(jv_source)

    uvvis_ref = register_raw_file(uvvis_source, MeasurementType.UV_VIS, dest_filename="uvvis.csv")
    ftir_ref = register_raw_file(ftir_source, MeasurementType.FTIR, dest_filename="ftir.csv")
    eis_ref = register_raw_file(eis_source, MeasurementType.EIS, dest_filename="eis.csv")
    jv_ref = register_raw_file(jv_source, MeasurementType.JV_CURVE, dest_filename="jv.csv")

    # --- create the RawSpectrum rows each raw file is referenced by ------------
    raw_spectrum_repo = RawSpectrumRepository(session)
    uvvis_spectrum = raw_spectrum_repo.add(
        RawSpectrum(
            measurement_type=MeasurementType.UV_VIS,
            pigment_extract_batch_id=pigment_batch.id,
            raw_data_reference=uvvis_ref.to_reference_string(),
            acquisition_date=date(2026, 1, 15),
            operator="J. Doe",
            instrument="Shimadzu UV-1800",
        )
    )
    ftir_spectrum = raw_spectrum_repo.add(
        RawSpectrum(
            measurement_type=MeasurementType.FTIR,
            pigment_extract_batch_id=pigment_batch.id,
            raw_data_reference=ftir_ref.to_reference_string(),
            acquisition_date=date(2026, 1, 15),
            operator="J. Doe",
            instrument="Nicolet iS10",
        )
    )
    eis_spectrum = raw_spectrum_repo.add(
        RawSpectrum(
            measurement_type=MeasurementType.EIS,
            device_id=device.id,
            raw_data_reference=eis_ref.to_reference_string(),
            acquisition_date=date(2026, 1, 20),
            operator="J. Doe",
            instrument="Autolab PGSTAT",
        )
    )
    jv_spectrum = raw_spectrum_repo.add(
        RawSpectrum(
            measurement_type=MeasurementType.JV_CURVE,
            device_id=device.id,
            raw_data_reference=jv_ref.to_reference_string(),
            acquisition_date=date(2026, 1, 20),
            operator="J. Doe",
            instrument="Keithley 2400",
        )
    )
    session.commit()

    # --- derive descriptors for all four modalities -----------------------------
    uvvis_descriptors = derive_uvvis_descriptors(
        uvvis_spectrum.id, uvvis_ref.to_reference_string(), operator="J. Doe"
    )
    ftir_descriptors = derive_ftir_descriptors(
        ftir_spectrum.id, ftir_ref.to_reference_string(), operator="J. Doe"
    )
    eis_measurement = derive_eis_measurement(
        device_id=device.id,
        source_spectrum_id=eis_spectrum.id,
        raw_data_reference=eis_ref.to_reference_string(),
        operator="J. Doe",
        instrument="Autolab PGSTAT",
    )
    jv_measurement = derive_jv_measurement(
        device_id=device.id,
        raw_data_reference=jv_ref.to_reference_string(),
        illumination_protocol=IlluminationProtocol.AM1_5G_1_SUN,
        measurement_date=date(2026, 1, 20),
        operator="J. Doe",
        instrument="Keithley 2400",
        raw_curve_id=jv_spectrum.id,
    )

    # --- persist everything ------------------------------------------------------
    descriptor_repo = DerivedDescriptorRepository(session)
    persisted_uvvis = [descriptor_repo.add(d) for d in uvvis_descriptors]
    persisted_ftir = [descriptor_repo.add(d) for d in ftir_descriptors]
    persisted_eis = ElectrochemicalMeasurementRepository(session).add(eis_measurement)
    persisted_jv = PhotovoltaicMeasurementRepository(session).add(jv_measurement)
    session.commit()

    # --- verify the full provenance chain, walkable end to end -----------------
    assert all(d.source_spectrum_id == uvvis_spectrum.id for d in persisted_uvvis)
    assert all(d.source_spectrum_id == ftir_spectrum.id for d in persisted_ftir)
    assert persisted_eis.raw_spectrum_id == eis_spectrum.id
    assert persisted_eis.device_id == device.id
    assert persisted_jv.raw_curve_id == jv_spectrum.id
    assert persisted_jv.device_id == device.id

    assert uvvis_spectrum.pigment_extract_batch_id == pigment_batch.id
    assert ftir_spectrum.pigment_extract_batch_id == pigment_batch.id
    assert eis_spectrum.device_id == device.id
    assert jv_spectrum.device_id == device.id
    assert pigment_batch.plant_sample_id == sample.id
    assert device.experimental_condition_id is not None  # -> species, walkable further

    # --- verify numeric results landed correctly (sanity, not re-derivation) ---
    uvvis_by_name = {d.descriptor_name: d.value for d in persisted_uvvis}
    assert uvvis_by_name["lambda_max_nm"] == pytest.approx(520.0, abs=0.5)

    ftir_by_name = {d.descriptor_name: d.value for d in persisted_ftir}
    assert ftir_by_name["ftir_peak_2_position"] == pytest.approx(1715.0, abs=1.0)

    assert persisted_eis.series_resistance_ohm == pytest.approx(15.0, rel=1e-2)
    assert persisted_jv.open_circuit_voltage_v == pytest.approx(0.68, abs=1e-3)

    # --- raw files are still intact and checksum-verified after the full run ---
    for reference in (uvvis_ref, ftir_ref, eis_ref, jv_ref):
        resolve_raw_file(reference.to_reference_string())  # raises if tampered/missing

    # --- append-only protection (Chunk 1.8) still covers what M2 writes to -----
    orm_row = session.get(RawSpectrumORM, uvvis_spectrum.id)
    orm_row.notes = "attempted edit"
    with pytest.raises(AppendOnlyViolationError):
        session.commit()
    session.rollback()
