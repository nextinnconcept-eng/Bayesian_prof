"""Turn rows the owner typed or uploaded into saved, validated real data.

Pipeline for one batch:

    raw rows (strings) -> prepare_rows()  fill safe defaults, tidy spelling,
                                          auto-name experiments and cells
                       -> check_rows()    dry run, no database writes: plain-language
                                          problems and warnings per row
                       -> import_past_results()
                                          ingest_rows() (all-or-nothing) then walk each
                                          experiment through the Section K states to
                                          INGESTED so it becomes training data

Rows saved here are ALWAYS real data: `is_synthetic` is forced to false, so
a user cannot accidentally (or deliberately) create simulated rows from the
form or an upload.

Approval: these are results of experiments the owner already ran, so the
owner is recorded as the approver, with a written note, as a human action.
Software never approves a real experiment on its own (state_machine.approve
refuses automation for real rows) -- this module passes
approved_by_automation=False and only runs when the owner presses the button.
"""

from __future__ import annotations

import hashlib
import io
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field

import pandas as pd
from sqlalchemy.orm import Session

from dssc_discovery.closed_loop import state_machine as sm
from dssc_discovery.closed_loop.real_ingestion import advance_measured_conditions
from dssc_discovery.database.repositories import (
    DeviceRepository,
    ExperimentalConditionRepository,
)
from dssc_discovery.domain.enums import ClosedLoopState, SpeciesCode
from dssc_discovery.ingestion.manual_entry import (
    IngestionReport,
    RowIngestionError,
    ingest_rows,
    parse_row,
)
from dssc_discovery.optimization.search_space import search_space_for
from dssc_discovery.user_data.field_guide import (
    ALL_FIELDS,
    FIELDS_BY_NAME,
    RECIPE_FIELD_NAMES,
    REQUIRED_FIELD_NAMES,
)
from dssc_discovery.validation.exceptions import DomainValidationError
from dssc_discovery.validation.rules import check_pce_consistency

OPEN_STATES = (ClosedLoopState.PROPOSED, ClosedLoopState.APPROVED)
_DONE_STATES = (
    ClosedLoopState.VALIDATED,
    ClosedLoopState.INGESTED,
    ClosedLoopState.MODEL_UPDATED,
)

_PCE_POWER = {"am1.5g_1_sun": 100.0, "am1.5g_half_sun": 50.0}


# --- reading an uploaded file ------------------------------------------------


class UnreadableFileError(ValueError):
    pass


def read_uploaded_table(content: bytes, filename: str) -> list[dict[str, str]]:
    """CSV or Excel bytes -> a list of string rows (blank cells become '')."""
    name = filename.lower()
    try:
        if name.endswith((".xlsx", ".xlsm")):
            frame = pd.read_excel(io.BytesIO(content), dtype=str, keep_default_na=False)
        elif name.endswith((".csv", ".txt")):
            frame = pd.read_csv(
                io.StringIO(content.decode("utf-8-sig")), dtype=str, keep_default_na=False
            )
        else:
            raise UnreadableFileError("Please upload a .csv or .xlsx file.")
    except UnreadableFileError:
        raise
    except Exception as exc:  # noqa: BLE001 - any parser failure is "unreadable" to a user
        raise UnreadableFileError(f"The file could not be read ({exc}).") from exc
    frame.columns = [str(c).strip() for c in frame.columns]
    rows = frame.to_dict(orient="records")
    return [r for r in rows if any(str(v).strip() for v in r.values())]


# --- normalising ---------------------------------------------------------------


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", text.strip().lower()).strip("_")


def _match_choice(name: str, raw: str) -> str:
    """Forgiving match of typed text to a stored choice value."""
    spec = FIELDS_BY_NAME[name]
    wanted = _slug(raw)
    for value, label in spec.choices:
        if wanted in {_slug(value), _slug(label)}:
            return value
    if name == "species_code":
        for species in SpeciesCode:
            if wanted == _slug(species.name):
                return species.value
    return raw.strip()


def _short_hash(parts: Iterable[object]) -> str:
    text = "|".join(str(p) for p in parts)
    return hashlib.sha1(text.encode()).hexdigest()[:8]  # noqa: S324 - not security related


def _recipe_key(row: Mapping[str, str]) -> str:
    return _short_hash(
        [row.get("species_code", "")] + [_number_key(row.get(n, "")) for n in RECIPE_FIELD_NAMES]
    )


def _number_key(raw: str) -> str:
    try:
        return f"{float(raw):.6g}"
    except (TypeError, ValueError):
        return str(raw).strip().lower()


def prepare_rows(
    rows: list[dict[str, str]], operator: str, session: Session | None = None
) -> list[dict[str, str]]:
    """Fill safe defaults and tidy spelling. Never invents a measured value:
    only names, bookkeeping fields and the (derivable) PCE."""
    prepared: list[dict[str, str]] = []
    seen_per_condition: dict[str, int] = {}
    for original in rows:
        row = {
            str(k).strip().lower(): ("" if v is None else str(v).strip())
            for k, v in original.items()
        }
        for spec in ALL_FIELDS:
            if spec.kind == "choice" and row.get(spec.name):
                row[spec.name] = _match_choice(spec.name, row[spec.name])
        # Fractions typed as percentages (62 instead of 0.62) are NOT guessed at:
        # the range check reports them with an explanation instead.

        species = row.get("species_code", "")
        abbreviation = "".join(w[0] for w in species.split()[:2]).upper() or "XX"
        key = _recipe_key(row)
        row.setdefault("condition_code", "")
        if not row["condition_code"]:
            row["condition_code"] = f"{abbreviation}-{key}"
        code = row["condition_code"]

        row.setdefault("photoanode_batch_code", "")
        if not row["photoanode_batch_code"]:
            row["photoanode_batch_code"] = (
                "PA-"
                + _short_hash(
                    _number_key(row.get(n, ""))
                    for n in ("tio2_thickness_um", "sintering_temperature_c", "sintering_time_min")
                )
                + "-"
                + _slug(row.get("surface_treatment", "x"))
            )

        seen_per_condition[code] = seen_per_condition.get(code, 0) + 1
        if not row.get("replicate_number"):
            offset = 0
            if session is not None:
                existing = ExperimentalConditionRepository(session).get_one(condition_code=code)
                if existing is not None:
                    offset = len(
                        DeviceRepository(session).filter(experimental_condition_id=existing.id)
                    )
            row["replicate_number"] = str(offset + seen_per_condition[code])
        if not row.get("device_code"):
            row["device_code"] = f"{code}-R{row['replicate_number']}"

        if not row.get("power_conversion_efficiency_percent"):
            computed = _computed_pce(row)
            if computed is not None:
                row["power_conversion_efficiency_percent"] = f"{computed:.4f}"

        row["design_source"] = row.get("design_source") or "manual"
        row["operator"] = operator.strip() or "owner"
        row["instrument"] = row.get("instrument") or "not recorded"
        row["is_synthetic"] = "false"  # user-entered data is always real
        prepared.append(row)
    return prepared


def _computed_pce(row: Mapping[str, str]) -> float | None:
    power = _PCE_POWER.get(row.get("illumination_protocol", ""))
    try:
        voc = float(row["open_circuit_voltage_v"])
        jsc = float(row["short_circuit_current_density_ma_cm2"])
        ff = float(row["fill_factor"])
    except (KeyError, ValueError):
        return None
    if power is None:
        return None
    return voc * jsc * ff / power * 100.0


# --- checking (dry run) ----------------------------------------------------------


@dataclass
class RowProblem:
    row_number: int
    message: str


@dataclass
class RowCheck:
    problems: list[RowProblem] = field(default_factory=list)
    warnings: list[RowProblem] = field(default_factory=list)
    missing_by_field: dict[str, list[int]] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.problems


_MISSING = re.compile(r"missing required field '(\w+)'")
_NOT_NUMBER = re.compile(r"field '(\w+)' is not a valid number: (.*)")
_NOT_DATE = re.compile(r"field '(\w+)' is not a valid ISO date")
_BAD_CHOICE = re.compile(r"'(.*)' is not a valid (\w+)")


def friendly_reason(reason: str) -> tuple[str, str | None]:
    """(plain message, missing field name if it was a missing value)."""
    if m := _MISSING.search(reason):
        spec = FIELDS_BY_NAME.get(m.group(1))
        if spec is None:
            return (f"'{m.group(1)}' is missing.", m.group(1))
        unit = f" ({spec.unit})" if spec.unit else ""
        return (f"'{spec.label}'{unit} is missing. {spec.help}", spec.name)
    if m := _NOT_NUMBER.search(reason):
        spec = FIELDS_BY_NAME.get(m.group(1))
        label = spec.label if spec else m.group(1)
        return (f"'{label}' must be a number, but {m.group(2)} was entered.", None)
    if m := _NOT_DATE.search(reason):
        spec = FIELDS_BY_NAME.get(m.group(1))
        label = spec.label if spec else m.group(1)
        return (f"'{label}' must be a date written like 2026-10-18.", None)
    if m := _BAD_CHOICE.search(reason):
        spec = FIELDS_BY_NAME.get(m.group(2))
        if spec is not None and spec.choices:
            options = ", ".join(v for v, _ in spec.choices)
            return (
                f"'{m.group(1)}' is not an option for '{spec.label}'. Choose one of: {options}.",
                None,
            )
    if "PCE" in reason and "does not match" in reason:
        return (
            "The efficiency (PCE) does not match Voc × Jsc × FF ÷ light power. Check the "
            "units (FF as a fraction such as 0.62, Jsc in mA/cm²) or leave the efficiency "
            "blank to have it calculated.",
            None,
        )
    if "already exists with different parameters" in reason or "reused with different" in reason:
        return (
            "This experiment name was already used for a recipe with different settings. "
            "If you changed something in the lab, give it a new name (for example add -B).",
            None,
        )
    if "matches existing condition" in reason:
        return (
            "This exact recipe is already saved under a different experiment name. "
            "Use the same experiment name to add another cell of that recipe.",
            None,
        )
    return (reason, None)


def check_rows(
    rows: list[dict[str, str]], operator: str, session: Session | None = None
) -> RowCheck:
    """Dry run: nothing is written. Collects every problem, not just the first."""
    result = RowCheck()
    prepared = prepare_rows(rows, operator, session)
    for number, row in enumerate(prepared, start=1):
        try:
            parsed = parse_row(row, number)
        except RowIngestionError as exc:
            message, missing = friendly_reason(exc.reason)
            result.problems.append(RowProblem(number, message))
            if missing:
                result.missing_by_field.setdefault(missing, []).append(number)
            continue
        try:
            check_pce_consistency(parsed.pv_measurement)
        except DomainValidationError as exc:
            result.problems.append(RowProblem(number, friendly_reason(str(exc))[0]))
            continue
        result.warnings.extend(_warnings(number, row, parsed.condition.target_parameters))
    return result


def _warnings(row_number: int, row: Mapping[str, str], vector: object) -> list[RowProblem]:
    warnings: list[RowProblem] = []
    species = SpeciesCode(row["species_code"])
    values = vector.model_dump(mode="json")  # type: ignore[attr-defined]
    space = search_space_for(species)
    for problem in space.violations(values):
        if "not on a" in problem:
            continue  # the sintering-time step only constrains optimizer proposals
        warnings.append(
            RowProblem(
                row_number,
                f"{problem}. This is outside the range the optimizer explores, so the row is "
                "saved but will have little influence on what it recommends.",
            )
        )
    ff = float(row["fill_factor"])
    if ff > 1:
        warnings.append(
            RowProblem(
                row_number, f"Fill factor {ff:g} is above 1. Enter it as a fraction (0.62, not 62)."
            )
        )
    return warnings


# --- saving ----------------------------------------------------------------------


@dataclass
class ImportResult:
    report: IngestionReport
    problems: list[RowProblem] = field(default_factory=list)
    conditions_ingested: list[str] = field(default_factory=list)
    blocked: dict[str, str] = field(default_factory=dict)

    @property
    def saved(self) -> bool:
        return self.report.committed and not self.blocked


def _bring_to_ingested(session: Session, code: str, operator: str, blocked: dict[str, str]) -> bool:
    repository = ExperimentalConditionRepository(session)
    condition = repository.get_one(condition_code=code)
    if condition is None:
        blocked[code] = "experiment was not found after saving"
        return False
    if condition.state in (ClosedLoopState.INGESTED, ClosedLoopState.MODEL_UPDATED):
        return True
    if condition.state == ClosedLoopState.PROPOSED:
        if not sm.history(session, condition.id):
            sm.record_proposal(session, condition, operator)
        sm.approve(
            session,
            condition.id,
            operator,
            "owner recorded this experiment's lab results",
            approved_by_automation=False,
        )
        condition = repository.get_one(condition_code=code)
    if condition is not None and condition.state == ClosedLoopState.APPROVED:
        sm.mark_fabricated(session, condition.id, operator, "owner recorded lab results")
    advance = advance_measured_conditions(session, [code], operator)
    if code in advance.ingested:
        return True
    reason = advance.blocked.get(code) or advance.skipped.get(code) or "could not be ingested"
    blocked[code] = reason
    return False


def import_past_results(
    session: Session, rows: list[dict[str, str]], operator: str
) -> ImportResult:
    """Save rows as real, ingested training data. All-or-nothing for the rows
    themselves; an experiment that cannot reach INGESTED (for example it was
    rejected earlier) is reported in `blocked`."""
    prepared = prepare_rows(rows, operator, session)
    report = ingest_rows(prepared, session)
    result = ImportResult(report=report)
    if not report.committed:
        for failure in report.failures:
            message, _ = friendly_reason(failure.reason)
            result.problems.append(RowProblem(failure.row_number, message))
        return result
    seen: list[str] = []
    for row in prepared:
        if row["condition_code"] not in seen:
            seen.append(row["condition_code"])
    for code in seen:
        if _bring_to_ingested(session, code, operator.strip() or "owner", result.blocked):
            result.conditions_ingested.append(code)
    session.commit()
    return result


def missing_required_summary(rows: list[dict[str, str]]) -> list[str]:
    """Which required columns are absent from an uploaded file entirely."""
    present = {str(k).strip().lower() for r in rows[:1] for k in r}
    return [
        FIELDS_BY_NAME[n].label
        for n in REQUIRED_FIELD_NAMES
        if n not in present and n != "power_conversion_efficiency_percent"
    ]
