"""M8 (Chunks 8.1-8.6) on an in-memory database with real (non-
synthetic) rows: initial DoE proposals, human approval, validated
ingestion, gated retraining, BO cycle from the logged ModelRun, and the
audit export."""

from __future__ import annotations

import json
from datetime import date

import pytest

from dssc_discovery.closed_loop import state_machine as sm
from dssc_discovery.closed_loop.orchestrator import propose_conditions
from dssc_discovery.closed_loop.real_ingestion import advance_measured_conditions
from dssc_discovery.closed_loop.real_workflow import (
    InsufficientRealDataError,
    recommend_real_cycle,
    refit_logged_surrogate,
    retrain_on_real_data,
)
from dssc_discovery.closed_loop.synthetic_objective import SyntheticObjective
from dssc_discovery.config import clear_config_cache
from dssc_discovery.database.repositories import DeviceRepository, PhotovoltaicMeasurementRepository
from dssc_discovery.domain.enums import (
    ClosedLoopState,
    DesignSource,
    IlluminationProtocol,
    SpeciesCode,
)
from dssc_discovery.domain.measurements import PhotovoltaicMeasurement
from dssc_discovery.ingestion.manual_entry import ingest_rows
from dssc_discovery.optimization.initial_design import latin_hypercube_design
from dssc_discovery.optimization.search_space import search_space_for
from dssc_discovery.reports.closed_loop_audit_export import export_audit_trail

SPECIES = SpeciesCode.BETA_VULGARIS
SPACE = search_space_for(SPECIES)


@pytest.fixture(autouse=True)
def small_threshold(monkeypatch, tmp_path):
    monkeypatch.setenv("DSSC_MIN_REAL_CONDITIONS_FOR_RETRAINING", "6")
    monkeypatch.setenv("DSSC_BO_BATCH_SIZE", "2")
    monkeypatch.setenv("DSSC_MLFLOW_TRACKING_URI", str(tmp_path / "mlruns"))
    clear_config_cache()
    yield
    clear_config_cache()


def _csv_row(condition, replicate, pce):
    v = condition.target_parameters.model_dump(mode="json")
    voc, ff = 0.65, 0.6
    return {
        "condition_code": condition.condition_code,
        "species_code": SPECIES.value,
        "design_source": condition.design_source.value,
        **{k: str(val) for k, val in v.items()},
        "extraction_temperature_c_unit": "",
        "photoanode_batch_code": f"{condition.condition_code}-PA",
        "device_code": f"{condition.condition_code}-R{replicate}",
        "replicate_number": str(replicate),
        "active_area_cm2": "0.25",
        "electrolyte_type": "iodide_triiodide",
        "counter_electrode_type": "platinum",
        "illumination_protocol": "am1.5g_1_sun",
        "measurement_date": "2026-10-01",
        "open_circuit_voltage_v": str(voc),
        "short_circuit_current_density_ma_cm2": str(pce / 100 * 100 / (voc * ff)),
        "fill_factor": str(ff),
        "power_conversion_efficiency_percent": str(pce),
        "operator": "Lab Tech",
        "instrument": "Keithley 2400",
        "is_synthetic": "false",
    }


def _run_lab(session, conditions, objective_seed=0):
    import numpy as np

    objective, rng = SyntheticObjective(SPACE, noise_std=0.1), np.random.default_rng(objective_seed)
    rows = []
    for c in conditions:
        sm.approve(session, c.id, "Dr A", "reviewed")
        sm.mark_fabricated(session, c.id, "Lab Tech")
        rows += [_csv_row(c, r, objective.evaluate(c.target_parameters, rng)) for r in (1, 2)]
    session.commit()
    report = ingest_rows(rows, session)
    assert report.all_succeeded, report.failures
    return advance_measured_conditions(session, [c.condition_code for c in conditions], "Lab Tech")


def _initial(session, n):
    conditions = propose_conditions(
        session,
        SPECIES,
        latin_hypercube_design(SPACE, n, seed=0),
        design_source=DesignSource.INITIAL_DOE,
        code_prefix="BV-DOE",
        operator="Dr A",
        is_synthetic=False,
    )
    session.commit()
    return conditions


def test_initial_doe_is_real_proposed_and_unapproved(session):
    conditions = _initial(session, 8)
    assert all(c.state == ClosedLoopState.PROPOSED and not c.is_synthetic for c in conditions)
    for c in conditions:
        assert SPACE.violations(c.target_parameters.model_dump(mode="json")) == []


def test_validated_results_reach_ingested_and_bad_data_is_blocked(session):
    good, bad = _initial(session, 2)
    report = _run_lab(session, [good])
    assert report.ingested == [good.condition_code]

    sm.approve(session, bad.id, "Dr A", "ok")
    sm.mark_fabricated(session, bad.id, "Lab Tech")
    device = DeviceRepository(session).filter(experimental_condition_id=good.id)[0]
    batch_id = device.photoanode_fabrication_batch_id
    from dssc_discovery.domain.device import Device

    bad_device = DeviceRepository(session).add(
        Device(
            experimental_condition_id=bad.id,
            photoanode_fabrication_batch_id=batch_id,
            device_code="BAD-R1",
            replicate_number=1,
            active_area_cm2=0.25,
            electrolyte_type=device.electrolyte_type,
            counter_electrode_type=device.counter_electrode_type,
            operator="Lab Tech",
        )
    )
    PhotovoltaicMeasurementRepository(session).add(
        PhotovoltaicMeasurement(
            device_id=bad_device.id,
            illumination_protocol=IlluminationProtocol.AM1_5G_1_SUN,
            measurement_date=date(2026, 10, 1),
            open_circuit_voltage_v=0.65,
            short_circuit_current_density_ma_cm2=10.0,
            fill_factor=0.6,
            power_conversion_efficiency_percent=9.5,
            operator="Lab Tech",
            instrument="Keithley",
        )
    )
    blocked = advance_measured_conditions(session, [bad.condition_code], "Lab Tech")
    assert bad.condition_code in blocked.blocked
    assert sm.history(session, bad.id)[-1].to_state == ClosedLoopState.MEASURED


def test_retraining_is_refused_below_the_threshold(session):
    _run_lab(session, _initial(session, 4))
    with pytest.raises(InsufficientRealDataError, match="need >= 6"):
        retrain_on_real_data(session, SPECIES, "Dr A")


def test_retrain_then_bo_cycle_then_audit_export(session, tmp_path):
    _run_lab(session, _initial(session, 8))
    surrogate = retrain_on_real_data(session, SPECIES, "Dr A", seed=3)
    session.commit()
    assert surrogate.model_run_id is not None

    batch = recommend_real_cycle(session, SPECIES, cycle_number=1, operator="Dr A")
    session.commit()
    assert len(batch.conditions) == 2
    assert all(
        not c.is_synthetic and c.design_source == DesignSource.BO_CYCLE_1 for c in batch.conditions
    )
    assert batch.bo_iteration.model_run_id == surrogate.model_run_id

    # New data without retraining -> refuses to recommend from a stale model.
    _run_lab(session, batch.conditions, objective_seed=1)
    session.commit()
    with pytest.raises(InsufficientRealDataError, match="dataset changed"):
        refit_logged_surrogate(session, SPECIES)

    manifest = export_audit_trail(session, tmp_path / "audit", scope="real_only")
    content = json.loads(manifest.read_text())
    assert content["row_counts"]["recommendations"] == 2
    assert content["row_counts"]["closed_loop_state_log"] > 0
    assert (tmp_path / "audit" / "experimental_conditions.csv").exists()
    synthetic = json.loads(
        export_audit_trail(session, tmp_path / "syn", scope="synthetic_only").read_text()
    )
    assert synthetic["row_counts"]["experimental_conditions"] == 0
