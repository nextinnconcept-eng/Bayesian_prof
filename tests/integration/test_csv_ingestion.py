"""Integration tests for the CSV/manual ingestion pipeline (Chunk 1.7) --
the earliest point the software accepts real lab data.

tests/fixtures/sample_devices.csv is clearly-marked dummy data (every row
has is_synthetic=true, which flows through into the persisted records) --
never real experimental results.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dssc_discovery.database.repositories import (
    DeviceRepository,
    ExperimentalConditionRepository,
    PhotoanodeFabricationBatchRepository,
    PhotovoltaicMeasurementRepository,
)
from dssc_discovery.ingestion.csv_loader import load_condition_device_pv_csv
from dssc_discovery.ingestion.manual_entry import ingest_rows, parse_row

FIXTURE_PATH = Path(__file__).resolve().parent.parent / "fixtures" / "sample_devices.csv"


def _base_row(**overrides: str) -> dict[str, str]:
    row = {
        "condition_code": "BV-COND-001",
        "species_code": "Beta vulgaris",
        "design_source": "initial_doe",
        "solvent": "ethanol",
        "solvent_ratio": "0.7",
        "ph": "5.5",
        "extraction_temperature_c": "25.0",
        "extraction_time_min": "60.0",
        "pigment_concentration_mm": "10.0",
        "adsorption_time_min": "120.0",
        "sensitization_concentration_mm": "0.5",
        "sensitization_temperature_c": "25.0",
        "tio2_thickness_um": "12.0",
        "sintering_temperature_c": "450.0",
        "sintering_time_min": "30.0",
        "surface_treatment": "none",
        "photoanode_batch_code": "PA-BV-001",
        "device_code": "BV-COND-001-R1",
        "replicate_number": "1",
        "active_area_cm2": "0.25",
        "electrolyte_type": "iodide_triiodide",
        "counter_electrode_type": "platinum",
        "illumination_protocol": "am1.5g_1_sun",
        "measurement_date": "2026-01-20",
        "open_circuit_voltage_v": "0.68",
        "short_circuit_current_density_ma_cm2": "13.5",
        "fill_factor": "0.62",
        "power_conversion_efficiency_percent": "5.6916",
        "operator": "Test Operator",
        "instrument": "Keithley 2400",
        "is_synthetic": "true",
    }
    row.update(overrides)
    return row


# --- end-to-end: the real fixture file --------------------------------------


def test_sample_csv_ingests_fully_end_to_end(session):
    report = load_condition_device_pv_csv(FIXTURE_PATH, session)

    assert report.committed is True
    assert report.all_succeeded
    assert report.total_rows == 3
    assert report.succeeded_rows == 3

    # 3 rows, but only 2 distinct conditions (2 replicate devices share
    # BV-COND-001) and 2 distinct photoanode batches -- dedup worked.
    conditions = ExperimentalConditionRepository(session).list()
    assert len(conditions) == 2
    batches = PhotoanodeFabricationBatchRepository(session).list()
    assert len(batches) == 2
    devices = DeviceRepository(session).list()
    assert len(devices) == 3
    pv_measurements = PhotovoltaicMeasurementRepository(session).list()
    assert len(pv_measurements) == 3

    # every ingested record is flagged synthetic (dummy fixture data)
    assert all(c.is_synthetic for c in conditions)
    assert all(d.is_synthetic for d in devices)


def test_sample_csv_deduped_condition_shares_id_across_replicates(session):
    load_condition_device_pv_csv(FIXTURE_PATH, session)

    device_repo = DeviceRepository(session)
    devices = {d.device_code: d for d in device_repo.list()}
    r1 = devices["BV-COND-001-R1"]
    r2 = devices["BV-COND-001-R2"]
    assert r1.experimental_condition_id == r2.experimental_condition_id
    assert r1.photoanode_fabrication_batch_id == r2.photoanode_fabrication_batch_id


def test_sample_csv_unit_conversion_applied(session):
    load_condition_device_pv_csv(FIXTURE_PATH, session)

    condition = ExperimentalConditionRepository(session).get_one(condition_code="CO-COND-001")
    # row entered 298.15 kelvin -- must be stored as 25.0 (canonical degC)
    assert condition.target_parameters.extraction_temperature_c == pytest.approx(25.0, abs=1e-6)


# --- malformed rows: reported, not silently dropped, no partial commit -----


def test_missing_required_field_is_reported_with_row_number(session):
    rows = [_base_row(), _base_row(device_code="BV-COND-001-R2", operator="")]
    report = ingest_rows(rows, session)

    assert report.committed is False
    assert len(report.failures) == 1
    assert report.failures[0].row_number == 2
    assert "operator" in report.failures[0].reason


def test_bad_unit_is_reported_with_row_number(session):
    rows = [_base_row(extraction_temperature_c_unit="bananas")]
    report = ingest_rows(rows, session)

    assert report.committed is False
    assert len(report.failures) == 1
    assert report.failures[0].row_number == 1
    assert (
        "bananas" in report.failures[0].reason.lower()
        or "unit" in report.failures[0].reason.lower()
    )


def test_unparseable_number_is_reported(session):
    rows = [_base_row(ph="not-a-number")]
    report = ingest_rows(rows, session)
    assert report.committed is False
    assert "ph" in report.failures[0].reason


def test_out_of_range_value_is_reported(session):
    rows = [_base_row(ph="20.0")]  # domain model rejects pH > 14
    report = ingest_rows(rows, session)
    assert report.committed is False
    assert len(report.failures) == 1


def test_one_bad_row_rolls_back_the_whole_batch(session):
    rows = [_base_row(), _base_row(device_code="BV-COND-001-R2", ph="not-a-number")]
    report = ingest_rows(rows, session)

    assert report.committed is False
    # the FIRST row was individually valid, but nothing should have landed
    assert ExperimentalConditionRepository(session).list() == []
    assert DeviceRepository(session).list() == []
    assert PhotovoltaicMeasurementRepository(session).list() == []


def test_condition_code_reused_with_different_parameters_is_rejected(session):
    rows = [
        _base_row(),
        _base_row(device_code="BV-COND-001-R2", ph="7.0"),  # same code, different recipe
    ]
    report = ingest_rows(rows, session)

    assert report.committed is False
    assert len(report.failures) == 1
    assert "different parameters" in report.failures[0].reason


def test_pce_inconsistent_with_voc_jsc_ff_is_rejected(session):
    rows = [_base_row(power_conversion_efficiency_percent="99.0")]  # nowhere near Voc*Jsc*FF/Pin
    report = ingest_rows(rows, session)
    assert report.committed is False
    assert len(report.failures) == 1


# --- parse_row unit tests ----------------------------------------------------


def test_parse_row_missing_field_raises_row_ingestion_error():
    from dssc_discovery.ingestion.manual_entry import RowIngestionError

    with pytest.raises(RowIngestionError) as exc_info:
        parse_row(_base_row(species_code=""), row_number=5)
    assert exc_info.value.row_number == 5


def test_parse_row_unknown_enum_value_raises_row_ingestion_error():
    from dssc_discovery.ingestion.manual_entry import RowIngestionError

    with pytest.raises(RowIngestionError):
        parse_row(_base_row(solvent="kerosene"), row_number=1)


def test_parse_row_succeeds_and_returns_linked_objects():
    parsed = parse_row(_base_row(), row_number=1)
    assert parsed.device.experimental_condition_id == parsed.condition.id
    assert parsed.device.photoanode_fabrication_batch_id == parsed.photoanode_batch.id
    assert parsed.pv_measurement.device_id == parsed.device.id
