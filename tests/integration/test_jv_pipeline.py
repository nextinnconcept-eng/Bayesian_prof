"""Integration test: a synthetic ideal-diode J-V curve registered via
chemistry.raw_storage (Chunk 2.1), processed end-to-end by
derive_jv_measurement (Chunk 2.5) -- including the automatic Chunk 1.6
PCE consistency check -- and persisted through the repository layer.
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pytest

from dssc_discovery.chemistry.jv import derive_jv_measurement
from dssc_discovery.chemistry.raw_storage import register_raw_file
from dssc_discovery.config import clear_config_cache
from dssc_discovery.database.repositories import PhotovoltaicMeasurementRepository
from dssc_discovery.domain.enums import IlluminationProtocol, MeasurementType, SpeciesCode
from tests.seed_helpers import seed_full_device_chain

JSC_TRUE, VOC_TRUE, N_IDEALITY, VT = 12.0, 0.68, 1.5, 0.02585
J0 = JSC_TRUE / (np.exp(VOC_TRUE / (N_IDEALITY * VT)) - 1)


def _diode_current(voltage):
    return JSC_TRUE - J0 * (np.exp(voltage / (N_IDEALITY * VT)) - 1)


def _write_jv_curve_csv(path, step=0.005):
    voltage = np.arange(0.0, 0.75 + step, step)
    current = _diode_current(voltage)
    lines = ["voltage_v,current_density_ma_cm2"]
    lines += [f"{v},{j}" for v, j in zip(voltage, current)]
    path.write_text("\n".join(lines), encoding="utf-8")


@pytest.fixture(autouse=True)
def _isolated_data_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("DSSC_DATA_DIR", str(tmp_path))
    clear_config_cache()
    yield
    clear_config_cache()


def test_full_pipeline_register_derive_persist(session, tmp_path):
    device = seed_full_device_chain(session, species_code=SpeciesCode.BETA_VULGARIS)
    source = tmp_path / "jv_curve.csv"
    _write_jv_curve_csv(source)
    reference = register_raw_file(source, MeasurementType.JV_CURVE, dest_filename="curve.csv")

    measurement = derive_jv_measurement(
        device_id=device.id,
        raw_data_reference=reference.to_reference_string(),
        illumination_protocol=IlluminationProtocol.AM1_5G_1_SUN,
        measurement_date=date(2026, 1, 20),
        operator="J. Doe",
        instrument="Keithley 2400",
    )

    assert measurement.open_circuit_voltage_v == pytest.approx(VOC_TRUE, abs=1e-3)
    assert measurement.short_circuit_current_density_ma_cm2 == pytest.approx(JSC_TRUE, abs=1e-6)
    assert 0.0 < measurement.fill_factor < 1.0
    assert measurement.power_conversion_efficiency_percent > 0.0

    repo = PhotovoltaicMeasurementRepository(session)
    persisted = repo.add(measurement)
    session.commit()

    refetched = repo.get(persisted.id)
    assert refetched is not None
    assert refetched.open_circuit_voltage_v == pytest.approx(VOC_TRUE, abs=1e-3)


def test_reproducible_given_same_curve(tmp_path):
    source = tmp_path / "jv_curve.csv"
    _write_jv_curve_csv(source)
    reference = register_raw_file(source, MeasurementType.JV_CURVE, dest_filename="curve.csv")

    from uuid import uuid4

    kwargs = dict(
        device_id=uuid4(),
        raw_data_reference=reference.to_reference_string(),
        illumination_protocol=IlluminationProtocol.AM1_5G_1_SUN,
        measurement_date=date(2026, 1, 20),
        operator="J. Doe",
        instrument="Keithley 2400",
    )
    first = derive_jv_measurement(**kwargs)
    second = derive_jv_measurement(**kwargs)

    assert first.open_circuit_voltage_v == second.open_circuit_voltage_v
    assert first.power_conversion_efficiency_percent == second.power_conversion_efficiency_percent
