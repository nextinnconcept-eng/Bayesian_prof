"""Row-ingestion engine: parses a wide-format record (one row = one
condition + photoanode fabrication batch + device + PV measurement) into
validated domain objects and persists them atomically.

Scope, deliberately: this covers the entities central to the DoE/BO loop
(ExperimentalCondition, PhotoanodeFabricationBatch, Device,
PhotovoltaicMeasurement) -- PV/PCE is literally the objective BO
optimizes. PlantSample/PigmentExtractBatch/SensitizationRun and raw
instrument data (RawSpectrum/DerivedDescriptor/ElectrochemicalMeasurement)
involve file references and belong with the characterization pipeline
(Chunk 2.x); they can get their own ingestion path there if needed,
following the same pattern.

Source-agnostic on purpose: this module knows nothing about CSV. The CSV
adapter (csv_loader.py) just turns file rows into dict[str, str] and
calls ingest_rows() -- a future manual-entry UI (Chunk 11.x) could call
the same function with a dict built from form fields.

A row's target_parameters and the PhotoanodeFabricationBatch it produces
are treated as equal to what was actually done -- this manual-entry path
has no separate "target vs. actual" distinction (that only matters once
BO proposes a target ahead of fabrication, Chunk 8.x); here, the lab
tech is recording what was actually executed, which serves as both.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

from pydantic import ValidationError as PydanticValidationError
from sqlalchemy.orm import Session

from dssc_discovery.database.repositories import (
    DeviceRepository,
    ExperimentalConditionRepository,
    PhotoanodeFabricationBatchRepository,
    PhotovoltaicMeasurementRepository,
    SpeciesRepository,
)
from dssc_discovery.domain.device import Device, PhotoanodeFabricationBatch
from dssc_discovery.domain.enums import (
    CounterElectrodeType,
    DesignSource,
    ElectrolyteType,
    IlluminationProtocol,
    SolventType,
    SpeciesCode,
    SurfaceTreatment,
)
from dssc_discovery.domain.experimental_condition import (
    ExperimentalCondition,
    ExperimentalConditionVector,
)
from dssc_discovery.domain.measurements import PhotovoltaicMeasurement
from dssc_discovery.domain.species import Species
from dssc_discovery.domain.units_registry import UnitConversionError, to_canonical
from dssc_discovery.logging_setup import get_logger
from dssc_discovery.validation.exceptions import DomainValidationError
from dssc_discovery.validation.rules import (
    check_duplicate_condition,
    check_pce_consistency,
    vectors_match,
)

logger = get_logger(__name__)

_VECTOR_UNIT_FIELDS = [
    "extraction_temperature_c",
    "extraction_time_min",
    "pigment_concentration_mm",
    "adsorption_time_min",
    "sensitization_concentration_mm",
    "sensitization_temperature_c",
    "tio2_thickness_um",
    "sintering_temperature_c",
    "sintering_time_min",
]


class RowIngestionError(Exception):
    """A single row failed to parse or validate. Carries the row number
    so a batch can report exactly which row and why, never silently
    dropping it.
    """

    def __init__(self, row_number: int, reason: str) -> None:
        self.row_number = row_number
        self.reason = reason
        super().__init__(f"row {row_number}: {reason}")


@dataclass
class RowFailure:
    row_number: int
    reason: str


@dataclass
class IngestionReport:
    total_rows: int
    failures: list[RowFailure] = field(default_factory=list)
    committed: bool = False

    @property
    def succeeded_rows(self) -> int:
        return self.total_rows - len(self.failures)

    @property
    def all_succeeded(self) -> bool:
        return not self.failures


@dataclass
class ParsedRow:
    condition: ExperimentalCondition
    photoanode_batch: PhotoanodeFabricationBatch
    device: Device
    pv_measurement: PhotovoltaicMeasurement


def _get_str(row: dict[str, str], field_name: str) -> str:
    value = (row.get(field_name) or "").strip()
    if not value:
        raise ValueError(f"missing required field {field_name!r}")
    return value


def _get_float(row: dict[str, str], field_name: str) -> float:
    raw = _get_str(row, field_name)
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"field {field_name!r} is not a valid number: {raw!r}") from exc


def _get_int(row: dict[str, str], field_name: str) -> int:
    raw = _get_str(row, field_name)
    try:
        return int(raw)
    except ValueError as exc:
        raise ValueError(f"field {field_name!r} is not a valid integer: {raw!r}") from exc


def _get_bool(row: dict[str, str], field_name: str, default: bool) -> bool:
    raw = (row.get(field_name) or "").strip().lower()
    if not raw:
        return default
    return raw in {"true", "1", "yes"}


def _get_date(row: dict[str, str], field_name: str) -> date:
    raw = _get_str(row, field_name)
    try:
        return date.fromisoformat(raw)
    except ValueError as exc:
        raise ValueError(
            f"field {field_name!r} is not a valid ISO date (YYYY-MM-DD): {raw!r}"
        ) from exc


def _get_unit_converted_float(row: dict[str, str], field_name: str) -> float:
    """Read a numeric field, converting from an optional '<field>_unit'
    override column to the field's canonical unit. If no override column
    is present, the value is assumed to already be in the canonical unit
    -- so a lab already recording in canonical units needs no extra
    columns at all.
    """
    value = _get_float(row, field_name)
    unit = (row.get(f"{field_name}_unit") or "").strip()
    if not unit:
        return value
    return to_canonical(value, unit, field_name)


def parse_row(row: dict[str, str], row_number: int) -> ParsedRow:
    """Parse and validate one wide-format row into domain objects, without
    touching the database. Raises RowIngestionError (with row_number
    attached) on any failure -- missing field, unparseable value, bad
    unit, or an out-of-range value the domain models themselves reject --
    never partially builds.
    """
    try:
        operator = _get_str(row, "operator")
        instrument = _get_str(row, "instrument")
        is_synthetic = _get_bool(row, "is_synthetic", default=False)

        vector = ExperimentalConditionVector(
            solvent=SolventType(_get_str(row, "solvent")),
            solvent_ratio=_get_float(row, "solvent_ratio"),
            ph=_get_float(row, "ph"),
            surface_treatment=SurfaceTreatment(_get_str(row, "surface_treatment")),
            **{name: _get_unit_converted_float(row, name) for name in _VECTOR_UNIT_FIELDS},
        )
        condition = ExperimentalCondition(
            species_code=SpeciesCode(_get_str(row, "species_code")),
            condition_code=_get_str(row, "condition_code"),
            design_source=DesignSource(_get_str(row, "design_source")),
            target_parameters=vector,
            operator=operator,
            is_synthetic=is_synthetic,
        )
        photoanode_batch = PhotoanodeFabricationBatch(
            batch_code=_get_str(row, "photoanode_batch_code"),
            tio2_thickness_um=vector.tio2_thickness_um,
            sintering_temperature_c=vector.sintering_temperature_c,
            sintering_time_min=vector.sintering_time_min,
            surface_treatment=vector.surface_treatment,
            operator=operator,
            is_synthetic=is_synthetic,
        )
        device = Device(
            experimental_condition_id=condition.id,
            photoanode_fabrication_batch_id=photoanode_batch.id,
            device_code=_get_str(row, "device_code"),
            replicate_number=_get_int(row, "replicate_number"),
            active_area_cm2=_get_unit_converted_float(row, "active_area_cm2"),
            electrolyte_type=ElectrolyteType(_get_str(row, "electrolyte_type")),
            counter_electrode_type=CounterElectrodeType(_get_str(row, "counter_electrode_type")),
            operator=operator,
            is_synthetic=is_synthetic,
        )
        pv_measurement = PhotovoltaicMeasurement(
            device_id=device.id,
            illumination_protocol=IlluminationProtocol(_get_str(row, "illumination_protocol")),
            measurement_date=_get_date(row, "measurement_date"),
            open_circuit_voltage_v=_get_unit_converted_float(row, "open_circuit_voltage_v"),
            short_circuit_current_density_ma_cm2=_get_unit_converted_float(
                row, "short_circuit_current_density_ma_cm2"
            ),
            fill_factor=_get_unit_converted_float(row, "fill_factor"),
            power_conversion_efficiency_percent=_get_unit_converted_float(
                row, "power_conversion_efficiency_percent"
            ),
            operator=operator,
            instrument=instrument,
            is_synthetic=is_synthetic,
        )
    except (ValueError, PydanticValidationError, UnitConversionError) as exc:
        raise RowIngestionError(row_number, str(exc)) from exc

    return ParsedRow(condition, photoanode_batch, device, pv_measurement)


def ingest_rows(rows: list[dict[str, str]], session: Session) -> IngestionReport:
    """Ingest a batch of wide-format rows atomically: if ANY row fails
    parsing or validation, NOTHING from this batch is committed -- the
    whole batch is rolled back and every failure is reported with its row
    number and reason. Rows sharing a condition_code or
    photoanode_batch_code (e.g. multiple replicate devices) reuse the
    same underlying record rather than creating duplicates, but only if
    their parameters actually match -- a code reused with different
    parameters is itself a reported failure, not silently accepted.
    """
    species_repo = SpeciesRepository(session)
    condition_repo = ExperimentalConditionRepository(session)
    photoanode_repo = PhotoanodeFabricationBatchRepository(session)
    device_repo = DeviceRepository(session)
    pv_repo = PhotovoltaicMeasurementRepository(session)

    failures: list[RowFailure] = []
    species_by_code: dict[SpeciesCode, Species] = {}
    condition_by_code: dict[str, ExperimentalCondition] = {}
    photoanode_by_code: dict[str, PhotoanodeFabricationBatch] = {}

    for row_number, row in enumerate(rows, start=1):
        try:
            parsed = parse_row(row, row_number)
            _get_or_create_species(
                parsed.condition.species_code,
                species_by_code,
                species_repo,
                operator=parsed.condition.operator,
                is_synthetic=parsed.condition.is_synthetic,
            )
            condition = _get_or_create_condition(
                parsed.condition, condition_by_code, condition_repo, row_number
            )
            photoanode_batch = _get_or_create_photoanode_batch(
                parsed.photoanode_batch, photoanode_by_code, photoanode_repo
            )

            device = device_repo.add(
                parsed.device.model_copy(
                    update={
                        "experimental_condition_id": condition.id,
                        "photoanode_fabrication_batch_id": photoanode_batch.id,
                    }
                )
            )

            pv_data = parsed.pv_measurement.model_copy(update={"device_id": device.id})
            try:
                check_pce_consistency(pv_data)
            except DomainValidationError as exc:
                raise RowIngestionError(row_number, str(exc)) from exc
            pv_repo.add(pv_data)

            logger.debug("row %d ingested (device %s)", row_number, device.device_code)
        except RowIngestionError as exc:
            failures.append(RowFailure(row_number=exc.row_number, reason=exc.reason))

    if failures:
        session.rollback()
        logger.warning("ingestion batch rolled back: %d/%d rows failed", len(failures), len(rows))
        return IngestionReport(total_rows=len(rows), failures=failures, committed=False)

    session.commit()
    logger.info("ingestion batch committed: %d rows", len(rows))
    return IngestionReport(total_rows=len(rows), failures=[], committed=True)


def _get_or_create_species(
    code: SpeciesCode,
    species_by_code: dict[SpeciesCode, Species],
    repo: SpeciesRepository,
    operator: str,
    is_synthetic: bool,
) -> Species:
    """experimental_conditions.species_code foreign-keys against
    species.code (Chunk 1.4) -- a condition can't be created until its
    species row exists, so ingestion has to ensure it, not just assume
    it's already there.
    """
    cached = species_by_code.get(code)
    if cached is not None:
        return cached

    existing = repo.get_one(code=code)
    result = (
        existing
        if existing is not None
        else repo.add(Species(code=code, operator=operator, is_synthetic=is_synthetic))
    )
    species_by_code[code] = result
    return result


def _get_or_create_condition(
    condition: ExperimentalCondition,
    condition_by_code: dict[str, ExperimentalCondition],
    repo: ExperimentalConditionRepository,
    row_number: int,
) -> ExperimentalCondition:
    code = condition.condition_code

    cached = condition_by_code.get(code)
    if cached is not None:
        if not vectors_match(cached.target_parameters, condition.target_parameters):
            raise RowIngestionError(
                row_number,
                f"condition_code {code!r} reused with different parameters within this batch",
            )
        return cached

    existing = repo.get_one(condition_code=code)
    if existing is not None:
        if not vectors_match(existing.target_parameters, condition.target_parameters):
            raise RowIngestionError(
                row_number, f"condition_code {code!r} already exists with different parameters"
            )
        condition_by_code[code] = existing
        return existing

    try:
        check_duplicate_condition(repo, condition)
    except DomainValidationError as exc:
        raise RowIngestionError(row_number, str(exc)) from exc

    created = repo.add(condition)
    condition_by_code[code] = created
    return created


def _get_or_create_photoanode_batch(
    batch: PhotoanodeFabricationBatch,
    batch_by_code: dict[str, PhotoanodeFabricationBatch],
    repo: PhotoanodeFabricationBatchRepository,
) -> PhotoanodeFabricationBatch:
    code = batch.batch_code

    cached = batch_by_code.get(code)
    if cached is not None:
        return cached

    existing = repo.get_one(batch_code=code)
    result = existing if existing is not None else repo.add(batch)
    batch_by_code[code] = result
    return result
