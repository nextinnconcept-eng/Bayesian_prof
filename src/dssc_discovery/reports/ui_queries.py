"""Read-only query helpers behind the Streamlit dashboard (M11): each
returns a plain DataFrame, so ui/ contains layout only and no business
logic, and the same tables can feed scripts and manuscript exports.
Every frame keeps is_synthetic so the UI can badge synthetic rows.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from uuid import UUID

import pandas as pd
from pydantic import BaseModel
from sqlalchemy.orm import Session

from dssc_discovery.database.repositories import (
    BOIterationRepository,
    ClosedLoopStateLogRepository,
    DerivedDescriptorRepository,
    DeviceRepository,
    ElectrochemicalMeasurementRepository,
    ExperimentalConditionRepository,
    HypothesisLogRepository,
    MLDatasetVersionRepository,
    ModelRunRepository,
    PhotovoltaicMeasurementRepository,
    PigmentExtractBatchRepository,
    PlantSampleRepository,
    RawSpectrumRepository,
    RecommendationRepository,
    SpeciesRepository,
    StabilityMeasurementRepository,
)


def _frame(rows: Sequence[BaseModel]) -> pd.DataFrame:
    return pd.DataFrame([r.model_dump(mode="json") for r in rows])


def conditions_frame(session: Session) -> pd.DataFrame:
    records = []
    for c in ExperimentalConditionRepository(session).list():
        record = c.model_dump(mode="json")
        record.update(record.pop("target_parameters"))
        records.append(record)
    return pd.DataFrame(records)


def library_frames(session: Session) -> dict[str, pd.DataFrame]:
    return {
        "species": _frame(SpeciesRepository(session).list()),
        "plant_samples": _frame(PlantSampleRepository(session).list()),
        "pigment_extract_batches": _frame(PigmentExtractBatchRepository(session).list()),
    }


def measurement_frames(session: Session) -> dict[str, pd.DataFrame]:
    return {
        "devices": _frame(DeviceRepository(session).list()),
        "photovoltaic": _frame(PhotovoltaicMeasurementRepository(session).list()),
        "raw_spectra": _frame(RawSpectrumRepository(session).list()),
        "derived_descriptors": _frame(DerivedDescriptorRepository(session).list()),
        "electrochemical": _frame(ElectrochemicalMeasurementRepository(session).list()),
        "stability": _frame(StabilityMeasurementRepository(session).list()),
    }


def model_runs_frame(session: Session) -> pd.DataFrame:
    frame = _frame(ModelRunRepository(session).list())
    if frame.empty:
        return frame
    metrics = frame["metrics_json"].map(json.loads).apply(pd.Series)
    return pd.concat([frame.drop(columns=["metrics_json"]), metrics.add_prefix("metric_")], axis=1)


def dataset_versions_frame(session: Session) -> pd.DataFrame:
    return _frame(MLDatasetVersionRepository(session).list())


def hypotheses_frame(session: Session) -> pd.DataFrame:
    return _frame(HypothesisLogRepository(session).list())


def recommendations_frame(session: Session) -> pd.DataFrame:
    """Recommendations joined to their condition's current state and code."""
    conditions = {c.id: c for c in ExperimentalConditionRepository(session).list()}
    iterations = {i.id: i for i in BOIterationRepository(session).list()}
    records = []
    for rec in RecommendationRepository(session).list():
        record = rec.model_dump(mode="json")
        condition = conditions.get(rec.experimental_condition_id)
        iteration = iterations.get(rec.bo_iteration_id)
        record["condition_code"] = condition.condition_code if condition else None
        record["state"] = condition.state.value if condition else None
        record["cycle_number"] = iteration.cycle_number if iteration else None
        record["species_code"] = iteration.species_code.value if iteration else None
        record["objective"] = iteration.objective if iteration else None
        records.append(record)
    frame = pd.DataFrame(records)
    if frame.empty:
        return frame
    leading = [
        "condition_code",
        "state",
        "species_code",
        "cycle_number",
        "objective",
        "rank",
        "predicted_objective",
        "predictive_uncertainty",
        "predicted_retention",
        "acquisition_value",
        "rationale_text",
    ]
    return frame[leading + [c for c in frame.columns if c not in leading]].sort_values(
        ["cycle_number", "rank"]
    )


def bo_iterations_frame(session: Session) -> pd.DataFrame:
    return _frame(BOIterationRepository(session).list())


def state_log_frame(session: Session) -> pd.DataFrame:
    codes = {c.id: c.condition_code for c in ExperimentalConditionRepository(session).list()}
    frame = _frame(ClosedLoopStateLogRepository(session).list())
    if not frame.empty:
        frame.insert(
            0,
            "condition_code",
            [codes.get(UUID(i)) for i in frame["experimental_condition_id"]],
        )
        frame = frame.sort_values("created_at")
    return frame


def audit_log_frame(session: Session, limit: int = 2000) -> pd.DataFrame:
    from dssc_discovery.database.audit import AuditLogORM

    rows = session.query(AuditLogORM).order_by(AuditLogORM.changed_at.desc()).limit(limit).all()
    return pd.DataFrame(
        [
            {
                "changed_at": r.changed_at,
                "table_name": r.table_name,
                "record_id": r.record_id,
                "operation": r.operation,
                "changed_fields": r.changed_fields,
            }
            for r in rows
        ]
    )
