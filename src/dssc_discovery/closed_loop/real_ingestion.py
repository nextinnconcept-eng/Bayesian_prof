"""Real result ingestion workflow (Chunk 8.2). After lab results are
loaded through the 1.7 ingestion path (ingestion/csv_loader.py, which
links rows to their PROPOSED-then-approved condition by condition_code),
advance_measured_conditions() walks each condition through
MEASURED -> VALIDATED -> INGESTED. Validation (1.6 rules) gates
VALIDATED: a failure leaves the condition MEASURED and reports why --
it never reaches the training data.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

from sqlalchemy.orm import Session

from dssc_discovery.closed_loop import state_machine as sm
from dssc_discovery.database.repositories import (
    DeviceRepository,
    ExperimentalConditionRepository,
    PhotovoltaicMeasurementRepository,
)
from dssc_discovery.domain.enums import ClosedLoopState
from dssc_discovery.validation.exceptions import DomainValidationError
from dssc_discovery.validation.rules import check_pce_consistency


@dataclass
class IngestionAdvanceReport:
    ingested: list[str] = field(default_factory=list)
    blocked: dict[str, str] = field(default_factory=dict)
    skipped: dict[str, str] = field(default_factory=dict)


def _validation_problems(session: Session, condition_id: object) -> list[str]:
    devices = DeviceRepository(session).filter(experimental_condition_id=condition_id)
    if not devices:
        return ["no devices recorded for this condition"]
    problems = []
    pv_repo = PhotovoltaicMeasurementRepository(session)
    measured = 0
    for device in devices:
        for measurement in pv_repo.filter(device_id=device.id):
            measured += 1
            try:
                check_pce_consistency(measurement)
            except DomainValidationError as exc:
                problems.append(f"{device.device_code}: {exc}")
    if measured == 0:
        problems.append("no photovoltaic measurements recorded")
    return problems


def advance_measured_conditions(
    session: Session, condition_codes: Iterable[str], operator: str
) -> IngestionAdvanceReport:
    """For each condition code: FABRICATED -> MEASURED (when PV data
    exists), then MEASURED -> VALIDATED -> INGESTED if validation passes.
    Caller must session.commit()."""
    report = IngestionAdvanceReport()
    repository = ExperimentalConditionRepository(session)
    for code in condition_codes:
        condition = repository.get_one(condition_code=code)
        if condition is None:
            report.skipped[code] = "unknown condition_code"
            continue
        if condition.is_synthetic:
            report.skipped[code] = "synthetic condition; real ingestion only handles real data"
            continue
        if condition.state == ClosedLoopState.FABRICATED:
            if _validation_problems(session, condition.id) == [
                "no devices recorded for this condition"
            ]:
                report.skipped[code] = "no measurements ingested yet"
                continue
            condition = sm.mark_measured(session, condition.id, operator, "lab results ingested")
        if condition.state != ClosedLoopState.MEASURED:
            report.skipped[code] = f"state is {condition.state.value}, expected fabricated/measured"
            continue
        problems = _validation_problems(session, condition.id)
        if problems:
            report.blocked[code] = "; ".join(problems)
            continue
        sm.mark_validated(session, condition.id, operator, "1.6 validation rules passed")
        sm.mark_ingested(session, condition.id, operator)
        report.ingested.append(code)
    return report
